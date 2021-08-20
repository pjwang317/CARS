import os
import time
import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import scipy.misc
import random
import subprocess
import cv2
from datetime import datetime

from modules.videosr_ops import *
from modules.utils import *
from modules.SSIM_Index import *
import modules.ps

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


### the mode should be "train", "test_K" or "test_pleiades"       
# mode = "test_WV3"
# mode = 'train'
mode = 'test_pleiades'
if mode == "test_pleiades":
    # test_path = 'D:/Dataset/Pleiades/test_pleiades30/'
    test_path = 'D:/Dataset/Pleiades/pls_test30/'
elif mode == "test_WV3":
    test_path = 'D:/Dataset/DFC2019/Train-Track1-RGB/test_input2/'

test_full = False

class SR(object):
    def __init__(self):
        self.num_frames = 1
        self.num_block = 10
        self.crop_size = 64
        self.scale_factor = 4

        self.max_steps = int(1.5e5+1)
        self.batch_size = 32
        self.eval_batch_size=32
        self.lstm_loss_weight = np.linspace(0.5, 1.0, self.num_frames)
        self.lstm_loss_weight = self.lstm_loss_weight / np.sum(self.lstm_loss_weight)
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.beta2=0.999
        self.decay_steps=3e3
        
        if mode == "test_WV3":
            self.train_dir = './checkpoint/DFC_SRCNNx4'
            self.save_path = './test_full/DFC_WV3_SRCNNx4/'
        elif mode == "test_pleiades":
            self.train_dir = './checkpoint/SRCNN_pleiadesx4'
            # self.save_path = './test_full/DFC_pleiades_SRCNNx4/'
            self.save_path = './test/pleiades_SRCNNx4_new/'
        elif mode == 'train':
            self.train_dir = './checkpoint/DFC_SRCNNx4'

        self.pathlist = open('./data/filelist_train.txt', 'rt').read().splitlines()
        random.shuffle(self.pathlist)
        self.vallist = open('./data/filelist_val.txt', 'rt').read().splitlines()



    def input_producer(self, batch_size=10):
        def read_data():
            idx0 = self.num_frames // 2
            data_seq = tf.random_crop(self.data_queue, [2, self.num_frames])
            input = tf.stack(
                [tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][idx0]), channels=3)])
            input, gt = prepprocessing(input, gt)
            print('Input producer shape: ', input.get_shape(), gt.get_shape())
            return input, gt

        def prepprocessing(input, gt=None):
            input = tf.cast(input, tf.float32) / 255.0
            gt = tf.cast(gt, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.crop_size, self.crop_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)
            #offset_gt = tf.concat([[0], offset[:2] * self.scale_factor, [0]], axis=-1)
            #size_gt = tf.concat([[1], size[:2] * self.scale_factor, [3]], axis=-1)
            offset_gt = tf.concat([[0], offset[:2] , [0]], axis=-1)
            size_gt = tf.concat([[1], size[:2] , [3]], axis=-1)
            gt = tf.slice(gt, offset_gt, size_gt)

            input.set_shape([self.num_frames, self.crop_size, self.crop_size, 3])
            #gt.set_shape([1, self.crop_size * self.scale_factor, self.crop_size * self.scale_factor, 3])
            gt.set_shape([1, self.crop_size , self.crop_size, 3])
            return input, gt

        with tf.variable_scope('input'):
            inList_all = []
            gtList_all = []
            for dataPath in self.pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'inputx{}/*.png'.format(self.scale_factor))))
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                inList_all.append(inList)
                gtList_all.append(gtList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=20)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=batch_size, num_threads=3, capacity=20)
        return batch_in, batch_gt

    def forward(self, frames_lr, is_training=True, reuse=False):
        num_batch, num_frame, height, width, num_channels = frames_lr.get_shape().as_list()
        out_height = height * self.scale_factor
        out_width = width * self.scale_factor
        idx0 = num_frame // 2
        frames_y = frames_lr
        frame_ref_y = frames_y[:, int(idx0), :, :, :]
        self.frames_y = frames_y
        self.frame_ref_y = frame_ref_y

        frame_bic_ref = tf.image.resize_images(frame_ref_y, [out_height, out_width], method=2)
        tf.summary.image('inp_0', im2uint8(frames_y[0, :, :, :, :]), max_outputs=3)
        tf.summary.image('bic', im2uint8(frame_bic_ref), max_outputs=3)

        x_unwrap = []



        for i in range(num_frame):
            if i > 0 and not reuse:
                reuse = True
            frame_i = frames_y[:, i, :, :, :]

            print('Build model - frame_{}'.format(i), frame_i.get_shape())
            frame_i_fw = frame_i

            with tf.variable_scope('srmodel', reuse=reuse) as scope_sr:#prelu
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                    biases_initializer=tf.constant_initializer(0.0)), \
                     slim.arg_scope([slim.batch_norm], center=True, scale=False, updates_collections=None,
                                    activation_fn=tf.nn.relu, epsilon=1e-5, is_training=is_training):
                    rnn_input = tf.concat([frame_i_fw], 3)
                    
                    filters = 64
                    conv0 = slim.conv2d(rnn_input, filters, [9, 9], scope='conv0')
                    conv1 = slim.conv2d(conv0, filters//2, [5, 5], scope='conv1')
                    conv2 = slim.conv2d(conv1, num_channels, [5, 5], scope='conv2')
                    
                    rnn_out = conv2

                if i >= 0:
                    x_unwrap.append(rnn_out)
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

        x_unwrap = tf.stack(x_unwrap, 1)
        return x_unwrap

    def build_model(self):
        frames_lr, frame_gt = self.input_producer(batch_size=self.batch_size)
        n, t, h, w, c = frames_lr.get_shape().as_list()
        output = self.forward(frames_lr)

        frame_gt_y = frame_gt
        
        #print('output shape :',output.shape)
        #print('y shape:',frame_gt_y)
        mse = tf.reduce_mean((output - frame_gt_y) ** 2, axis=[0, 2, 3, 4])
        self.mse = mse
        for i in range(self.num_frames):
            tf.summary.scalar('mse_%d' % i, mse[i])
        tf.summary.image('out_0', im2uint8(output[0, :, :, :, :]), max_outputs=3)
        tf.summary.image('res', im2uint8(output[:, -1, :, :, :]), max_outputs=3)
        tf.summary.image('gt', im2uint8(frame_gt_y[:, 0, :, :, :]), max_outputs=3)

        self.loss_mse = tf.reduce_sum(mse * self.lstm_loss_weight)
        tf.summary.scalar('loss_mse', self.loss_mse)

        self.loss = self.loss_mse
        tf.summary.scalar('loss_all', self.loss)

    def evaluation(self):
        print('Evaluating ...')
        inList_all = []
        gtList_all = []
        for dataPath in self.vallist:
            #inList = sorted(glob.glob(os.path.join(dataPath, 'input{}/*.png'.format(self.scale_factor))))
            inList = sorted(glob.glob(os.path.join(dataPath, 'inputx{}/*.png'.format(self.scale_factor))))
            gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
            inList_all.append(inList)
            gtList_all.append(gtList)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)
        sess = self.sess

        # out_h = 528
        # out_w = 960
        #out_h = 516
        #out_w = 640
        
        out_h = 256
        out_w = 256
        # in_h = out_h // self.scale_factor
        # in_w = out_w // self.scale_factor
        in_h = out_h
        in_w = out_w
        
        if not hasattr(self, 'eval_input'):
            self.eval_input = tf.placeholder(tf.float32, [self.eval_batch_size, self.num_frames, in_h, in_w, 3])
            self.eval_gt = tf.placeholder(tf.float32, [self.eval_batch_size, 1, out_h, out_w, 3])
            self.eval_output = self.forward(self.eval_input, is_training=False, reuse=True)

            # calculate loss
            frame_gt_y = self.eval_gt
            self.eval_mse = tf.reduce_mean((self.eval_output[:, :, :, :, :] - frame_gt_y) ** 2, axis=[2, 3, 4])

        batch_in = []
        batch_gt = []
        radius = self.num_frames // 2
        mse_acc = None
        ssim_acc = None
        batch_cnt = 0
        #batch_name=[]
        for inList, gtList in zip(inList_all, gtList_all):
            for idx0 in range(self.num_frames//2, len(inList), 6):
                #batch_name.append(gtList[idx0])
                inp = [scipy.misc.imread(inList[0]) for i in range(idx0 - radius, 0)]
                inp.extend([scipy.misc.imread(inList[i]) for i in range(max(0, idx0 - radius), idx0)])
                inp.extend([scipy.misc.imread(inList[i]) for i in range(idx0, min(len(inList), idx0 + radius + 1))])
                inp.extend([scipy.misc.imread(inList[-1]) for i in range(idx0 + radius, len(inList) - 1, -1)])
                inp = [i[:in_h, :in_w, :].astype(np.float32) / 255.0 for i in inp]
                #inp = sess.run(tf.image.resize_images(i, [out_h, out_w], method=2) for i in inp)
                #print('inp shape:',inp[0].shape)
                gt = [scipy.misc.imread(gtList[idx0])]
                gt = [i[:out_h, :out_w, :].astype(np.float32) / 255.0 for i in gt]

                batch_in.append(np.stack(inp, axis=0))
                batch_gt.append(np.stack(gt, axis=0))

                if len(batch_in) == self.eval_batch_size:
                    batch_cnt += self.eval_batch_size
                    batch_in = np.stack(batch_in, 0)
                    batch_gt = np.stack(batch_gt, 0)
                    #batch_in = sess.run(tf.image.resize_images(batch_in, [out_h, out_w], method=2))
                    #print('batch_in shape:', batch_in.shape)
                    #print('batch_gt shape:', batch_gt.shape)
                    mse_val, eval_output_val = sess.run([self.eval_mse, self.eval_output],
                                                        feed_dict={self.eval_input: batch_in, self.eval_gt: batch_gt})
                    ssim_val = np.array(
                        [[compute_ssim(eval_output_val[ib, it, :, :, 0], batch_gt[ib, 0, :, :, 0], l=1.0)
                          for it in range(self.num_frames)] for ib in range(self.eval_batch_size)])
                    if mse_acc is None:
                        mse_acc = mse_val
                        ssim_acc = ssim_val
                    else:
                        mse_acc = np.concatenate([mse_acc, mse_val], axis=0)
                        ssim_acc = np.concatenate([ssim_acc, ssim_val], axis=0)
                    batch_in = []
                    batch_gt = []
                    print('\tEval batch {} - {} ...'.format(batch_cnt, batch_cnt + self.eval_batch_size))

        psnr_acc = 10 * np.log10(1.0 / mse_acc)
        mse_avg = np.mean(mse_acc, axis=0)
        psnr_avg = np.mean(psnr_acc, axis=0)
        ssim_avg = np.mean(ssim_acc, axis=0)
        for i in range(mse_avg.shape[0]):
            tf.summary.scalar('val_mse{}'.format(i), tf.convert_to_tensor(mse_avg[i], dtype=tf.float32))
        print('Eval MSE: {}, PSNR: {},SSIM :{}'.format(mse_avg, psnr_avg,ssim_avg))
        # write to log file
        with open(os.path.join(self.train_dir, 'eval_log.txt'), 'a+') as f:
            f.write('Iter {} - MSE: {}, PSNR: {}, SSIM: {}\n'.format(sess.run(self.global_step), mse_avg, psnr_avg,
                                                                     ssim_avg))
        np.save(os.path.join(self.train_dir, 'eval_iter_{}'.format(sess.run(self.global_step))),
                {'mse': mse_acc, 'psnr': psnr_acc, 'ssim': ssim_acc})

    def train(self):
        def train_op_func(loss, var_list, is_gradient_clip=False):
            if is_gradient_clip:
                train_op = tf.train.AdamOptimizer(lr, self.beta1)
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=var_list, global_step=global_step)
            return train_op

        """Train sr network"""
        global_step = tf.Variable(initial_value=0, trainable=False)
        self.global_step = global_step

        # Create folder for logs
        if not tf.gfile.Exists(self.train_dir):
            tf.gfile.MakeDirs(self.train_dir)

        self.build_model()
        decay_steps = 5e3
        lr=tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, decay_rate=0.5, staircase=False)+1e-4
        tf.summary.scalar('learning_rate', lr)
        vars_all = tf.trainable_variables()
        vars_sr = [v for v in vars_all if 'srmodel' in v.name]
        train_all = train_op_func(self.loss, vars_all, is_gradient_clip=True)
        train_sr = train_op_func(self.loss_mse, vars_sr, is_gradient_clip=True)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)
        self.sess = sess
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            print(sess.run(tf.global_variables_initializer()))
        except Exception as e:
                #Report exceptions to the coordinator
            coord.request_stop(e)
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        self.load(sess, os.path.join(self.train_dir, 'checkpoints'))

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in range(sess.run(global_step), self.max_steps):
            if step < 10000:
                train_op = train_sr
            elif step < 20000:
                train_op = train_sr
            else:
                train_op = train_sr

            start_time = time.time()
            _, loss_value, mse_value, loss_mse_value = sess.run(
                [train_op, self.loss, self.mse, self.loss_mse])
            duration = time.time() - start_time + 0.01
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f: %.5f), mse = %s  (%.1f data/s; %.3f '
                              's/bch)')
                print((format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_value, loss_mse_value,
                                      str(mse_value), examples_per_sec, sec_per_batch)))
                with open(os.path.join(self.train_dir, 'train_log.txt'), 'a+') as f:
                    f.write('Iter {} - loss:{} , MSE: {}\n'.format(step, loss_value, loss_mse_value
                                                                     ))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            if step % 500 == 0:
                self.evaluation()
            if step % 500 == 499 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "videoSR.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading SR checkpoints...")
        model_name = "videoSR.model"

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading checkpoints...{} Success".format(ckpt_name))
            return True
        else:
            print(" [*] Reading checkpoints... There is no checkpoint")
            return False


    def test(self, dataPath=None, scale_factor=4, num_frames=1):

        import scipy.misc
        import math
        import time
        
        save_path =self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        
        file=os.listdir(dataPath)
        m = len(file)
        psnr_acc = np.zeros([len(file),2])
        ssim_acc = np.zeros([len(file),2])
        
        file=[os.path.join(dataPath,f) for f in file]
        file=[f for f in file if os.path.isdir(f)]
        startid=0
        cnt=0
        
        for pat in file:
            if file.index(pat)>=startid:
                dataPath = pat
                if test_full:
                    # inList = sorted(glob.glob(os.path.join(dataPath, 'inputx{}/*.png').format(scale_factor)))
                    inList = sorted(glob.glob(os.path.join(dataPath, 'inputup{}/*.png').format(scale_factor)))
                    inList_ref = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                    inp = [scipy.misc.imread(i)[1600:2400,1600:2400,:].astype(np.float32) / 255.0 for i in inList]
                    inp_ref = [scipy.misc.imread(i).astype(np.float32)/ 255.0 for i in inList_ref]
                else:
                    inList = sorted(glob.glob(os.path.join(dataPath, 'inputx{}/*.png').format(scale_factor)))
                    # inList = sorted(glob.glob(os.path.join(dataPath, 'inputup{}/*.png').format(scale_factor)))
                    inList_ref = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                    inp = [scipy.misc.imread(i).astype(np.float32) / 255.0 for i in inList]
                    inp_ref = [scipy.misc.imread(i).astype(np.float32)/ 255.0 for i in inList_ref]
                
                
                print('Testing path: {}'.format(dataPath))
                print('# of testing frames: {}'.format(len(inList)))

        
                print(inp[0].shape)
        
                self.scale_factor = scale_factor
                if file.index(pat)==startid:
                    reuse = False
                else:
                    reuse=True

                for idx0 in range(len(inList)):
                    cnt += 1
                    T = num_frames // 2

                    imgs = [inp[0] for i in np.arange(idx0 - T, 0)]
                    imgs.extend([inp[i] for i in np.arange(max(0, idx0 - T), idx0)])
                    imgs.extend([inp[i] for i in np.arange(idx0, min(len(inList), idx0 + T + 1))])
                    imgs.extend([inp[-1] for i in np.arange(idx0 + T, len(inList) - 1, -1)])

                    dims = imgs[0].shape
                    if len(dims) == 2:
                        imgs = [np.expand_dims(i, -1) for i in imgs]
                    h, w, c = imgs[0].shape
                    
                    out_h = h 
                    out_w = w 
                    padh = int(math.ceil(h / 4.0) * 4.0 - h)
                    padw = int(math.ceil(w / 4.0) * 4.0 - w)
                    imgs = [np.pad(i, [[0, padh], [0, padw], [0, 0]], 'edge') for i in imgs]
                    imgs = np.expand_dims(np.stack(imgs, axis=0), 0)

                    if idx0 == self.num_frames//2:
                        frames_lr = tf.placeholder(dtype=tf.float32, shape=imgs.shape)
                        frames_ref_ycbcr = frames_lr[:, T:T + 1, :, :, :]
                        frames_ref_ycbcr = tf.tile(frames_ref_ycbcr, [1, num_frames, 1, 1, 1])
                        output = self.forward(frames_lr, is_training=False, reuse=reuse)
                        # print (frames_lr_ycbcr.get_shape(), h, w, padh, padw)
                        output_rgb = output
                        output = output[:, :, :out_h, :out_w, :]
                        output_rgb = output_rgb[:, :, :out_h, :out_w, :]

                    if cnt == 1:
                        config = tf.ConfigProto()
                        config.gpu_options.allow_growth = True
                        sess = tf.Session(config = config)
                        # sess = tf.Session()
                        reuse = True
                        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
                        self.load(sess, os.path.join(self.train_dir, 'checkpoints'))
                    case_path = dataPath.split('/')[-1]
                    print('Testing - ', case_path, len(imgs))
                    
                    bic_ref = sess.run(tf.image.resize_images(inp[0], [out_h, out_w], method=2))
                    st_time=time.time()
                    
                    [imgs_hr_rgb] = sess.run([output_rgb],feed_dict={frames_lr: imgs})
                    
                    ed_time=time.time()
                    cost_time=ed_time-st_time
                    print('spent {} s.'.format(cost_time))
                    
                    # psnr_SRCNN = sess.run(tf.image.psnr(inp_ref[0],imgs_hr_rgb[0,-1],max_val=1.0))
                    # psnr_bicubic = sess.run(tf.image.psnr(inp_ref[0],bic_ref,max_val=1.0))
                    # #psnr_PECNN = 10 * np.log10(1.0 / mse_acc)
                    # psnr_acc[cnt-1,0] = psnr_SRCNN
                    # psnr_acc[cnt-1,1] = psnr_bicubic
                    # print('psnr_SRCNN:',psnr_SRCNN, 'psnr_bicubic', psnr_bicubic)
                    # inp_ref_ = tf.convert_to_tensor(inp_ref[0], dtype = tf.float32)
                    # imgs_hr_rgb_ = tf.convert_to_tensor(imgs_hr_rgb[0,-1], dtype = tf.float32)
                    # bic_ref_ = tf.convert_to_tensor(bic_ref,dtype = tf.float32)
                    # ssim_bicubic = sess.run( tf.image.ssim(inp_ref_,bic_ref_,max_val=1.0))
                    # ssim_SRCNN = sess.run( tf.image.ssim(inp_ref_,imgs_hr_rgb_,max_val=1.0))
                    # ssim_acc[cnt-1,0] = ssim_SRCNN
                    # ssim_acc[cnt-1,1] = ssim_bicubic
                    # print('ssim_SRCNN:',ssim_SRCNN,'ssim_bicubic',ssim_bicubic)
                    # with open(os.path.join(self.train_dir, 'test_log.txt'), 'a+') as f:
                        # f.write('Img num {0:3d} -  PSNR_SRCNN: {1:0.6f}, PSNR_bic:{2:0.6f},SSIM_SRCNN: {3:0.6f},SSIM_bic:{4:0.6f}\n'.format(cnt, psnr_SRCNN, psnr_bicubic, ssim_SRCNN,ssim_bicubic))

                    if len(dims) == 3 and idx0==self.num_frames//2:
                        scipy.misc.imsave(os.path.join(save_path, 'rgb_%03d.png' % (file.index(pat))),
                                  im2uint8(imgs_hr_rgb[0, -1, :, :, :]))
                        # scipy.misc.imsave(os.path.join(save_path,'bic_rgb_%03d.png' %(file.index(pat))),im2uint8(bic_ref))
                #print('SR results path: {}'.format(save_path))
                

                
            else:
                print('startid error')
                
        # psnr_avg = np.sum(psnr_acc,axis=0)/m
        # ssim_avg = np.sum(ssim_acc,axis=0)/m
        # print('psnr_SRCNN_average:',psnr_avg[0],'psnr_bicubic_average:',psnr_avg[1])
        # print('ssim_SRCNN_average:',ssim_avg[0],'ssim_bicubic_average:',ssim_avg[1])
        # with open(os.path.join(self.train_dir, 'test_log.txt'), 'a+') as f:
            # f.write('Average:- PSNR_SRCNN_avg: {0:0.6f}, PSNR_bic_avg:{1:0.6f},SSIM_SRCNN_avg: {2:0.6f},SSIM_bic_avg:{3:0.6f}\n'.format(psnr_avg[0],
                                                                                        # psnr_avg[1], ssim_avg[0],ssim_avg[1]))

def main(_):

    # st_time=time.time()
    model = SR()
    if mode == "train":
        model.train()
    elif mode == "test_WV3":
    #model.evaluation()
        model.test(test_path)
    elif mode == "test_pleiades":
        model.test(test_path)
    else:
        print("Please select the right mode!")
    # model.train()
    #model.evaluation()
    # model.test('D:\\Dataset\\Kaggle_RS\\Kaggle_test\\test30\\')
    # model.test('D:\\super-resolution\\Modified_codes\\11-PECNN\PECNN-master\\test\\test_pleiades30\\')
    # ed_time=time.time()
    # cost_time=ed_time-st_time
    # print('spent {} s.'.format(cost_time))


if __name__ == '__main__':
    tf.app.run()
