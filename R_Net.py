#sudo fuser -v /dev/nvidia*

import numpy as np
import tensorflow as tf
from data_preprocess import R_Net_Data
import logging
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("seq2seq")

class Config(object):
    '''
    默认配置
    '''
    learning_rate=0.001
    batch_size=168
    Q_len=15    # 问句长度
    P_len=100    # 文档长度
    embedding_dim=50    #词向量维度
    hidden_dim=100
    train_dir='./data/train_out_500.txt'
    dev_dir='./data/dev_out.txt'
    test_dir='./data/test.txt'
    model_dir='./save_model/r_net_model_500.ckpt'
    # train_num=50
    use_cpu_num=8
    keep_dropout=0.7
    summary_write_dir="./tmp/r_net.log"
    epoch=1000

config=Config()
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_float("keep_dropout", config.keep_dropout, "dropout")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("Q_len", config.Q_len, "问句长度")
tf.app.flags.DEFINE_integer("P_len", config.P_len, "文档长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入维度.")
tf.app.flags.DEFINE_integer("hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("epoch", config.epoch, "每轮训练迭代次数")
tf.app.flags.DEFINE_string("summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_string("mod", "train", "默认为训练") # true for prediction
FLAGS = tf.app.flags.FLAGS


class R_Net(object):
    '''
    R_net 智能问答模型
    '''
    def __init__(self,embedding_dim,hidden_dim,Q_len,P_len,batch_size,vocab_size,keep_drouput=1.0):
        self.embedding_dim=embedding_dim
        self.Q_len=Q_len
        self.P_len=P_len
        self.num_class=5
        self.hidden_dim=hidden_dim
        self.keep_dropout=keep_drouput
        with tf.device("/cpu:0"):
            self.embedding=tf.Variable(tf.random_normal(shape=(vocab_size,self.embedding_dim),mean=0.0,stddev=1.0)) #词向量矩阵

            self.Q=tf.placeholder(dtype=tf.int32,shape=(None,self.Q_len))
            self.P=tf.placeholder(dtype=tf.int32,shape=(None,self.P_len))

            self.Q_array=tf.nn.embedding_lookup(self.embedding,self.Q)  #将问句 词id转换为词矩阵
            self.P_array=tf.nn.embedding_lookup(self.embedding,self.P)  #将文档 词id转换为词矩阵

            self.Q_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,)) # 问句实际长度
            self.P_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,)) # 文档实际长度
            self.out_num=2 #输出数量 为2 则输出为 起始位置和结束位置
            self.label = tf.placeholder(dtype=tf.int32, shape=(None, self.out_num))
        #self.batch_size=self.Q.get_shape().as_list()[0]
        self.batch_size=batch_size

        with tf.device("/gpu:0"):
            Q_,_ = self.process(input_=self.Q_array,seq_len=self.Q_len,seq_vec=self.Q_seq_vec,scope="Q_encoder")
            P_,_ = self.process(input_=self.P_array,seq_len=self.P_len,seq_vec=self.P_seq_vec,scope="P_encoder")
            Q_1=tf.contrib.layers.batch_norm(Q_) #批标准化
            P_1=tf.contrib.layers.batch_norm(P_)
        # gated_attention
            P_2=self.gated_attention(Q_1,P_1) #[None,P_len,hidden_dim]
            P_2=tf.contrib.layers.batch_norm(P_2)
            # self.P_=tf.nn.dropout(self.P_,self.keep_dropout)
            # self_match_attention
            H=self.self_match_attention(P_2)
            H_1=tf.contrib.layers.batch_norm(H)
            self.soft_logits,self.ids=self.pointer_network(Q_1,H_1)

        label_one_hot = tf.one_hot(self.label, self.P_len, 1, 0, 2)
        # logit=tf.reshape(self.soft_logits,(-1,self.out_num,self.P_len))
        logit=self.soft_logits
        logit1=tf.clip_by_value(logit,1e-5,1.0) #截断函数 防止nan 溢出
        logit1=tf.contrib.layers.batch_norm(logit1)
        with tf.device('/gpu:0'):
            self.loss=tf.losses.softmax_cross_entropy(logits=logit1,onehot_labels=label_one_hot,weights=1.0)
            tf.summary.scalar("loss",self.loss)
            self.optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.merge_summary=tf.summary.merge_all()

    def process(self,input_,seq_len,seq_vec,scope):
        '''
        preocess layer
        :return: 
        '''
        state=[]
        with tf.variable_scope(scope):
            lstm_input=tf.transpose(input_,[1,0,2])
            lstm_input = tf.unstack(lstm_input, seq_len, 0)
            cell_f = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)

            cell_b = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)
            #static_bidirectional_rnn
            (out,fw_state,_)=tf.contrib.rnn.static_bidirectional_rnn(cell_f,cell_b,lstm_input,dtype=tf.float32,
                                                               sequence_length=seq_vec)
            state.append(fw_state)
            out=tf.stack(out,0)
            out=tf.transpose(out,[1,0,2])
            return out,state[-1]

    def gated_attention(self,Q_,P_):
        '''
        gated attention模块
        :return: RNN的输出结果
        '''
        P_P = tf.transpose(P_, [1, 0, 2])
        P_list=tf.unstack(P_P,self.P_len,0)
        #init_c=tf.zeros(shape=(self.batch_size,self.hidden_dim))
        init_v=tf.zeros(shape=(self.batch_size,self.hidden_dim))
        #C=[init_c]

        lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                       state_is_tuple=True)
        V = [init_v]
        #C = [init_v]
        init_state=lstm_cell.zero_state(self.batch_size, tf.float32)
        #print("init_state",init_state)
        state=[init_state]
        with tf.variable_scope("gated_attention"):
            for t in range(self.P_len):
                if t>0:
                    tf.get_variable_scope().reuse_variables()
                w_g = tf.Variable(tf.random_normal(shape=(4 * self.hidden_dim, 4 * self.hidden_dim)))

                u_P_t=tf.reshape(P_list[t],[-1,1,2*self.hidden_dim])
                v_t_=tf.reshape(V[-1],[-1,1,self.hidden_dim])

                c_t=self.gated_attention_ops(Q_,u_P_t=u_P_t,v_t_=v_t_)#[None,2*hidden_dim]
                u_p_t_c_t=tf.concat((P_list[t],c_t),1)
                u_p_t_c_t_=tf.sigmoid(tf.matmul(u_p_t_c_t,w_g))*u_p_t_c_t
                #v
                (v_t,new_state)=lstm_cell(u_p_t_c_t_,state[-1])# v_t=[None,self.hidden_dim]

                V.append(v_t)
                state.append(new_state)
            out=V[1::]
            out=tf.stack(out)
            out=tf.transpose(out,[1,0,2])#[None,P_len,hidden_dim]
            return out

    def gated_attention_ops(self,u_Q,u_P_t,v_t_):
        '''
        u_Q=[None,Q_len,2*hidden_dim]
        u_P=[None,1,2*hidden_dim]
        v_t_=[None,1,hidden_dim]
        :param u_Q: 
        :param u_P: 
        :param v_t_: 
        :return: 
        '''
        with tf.device('/gpu:0'):
            with tf.variable_scope("atteion",reuse=True):
                w_u_Q = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.hidden_dim)))
                w_u_P = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.hidden_dim)))
                w_v_P = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
                V = tf.Variable(tf.random_normal(shape=(self.hidden_dim,1)))

                QQ=tf.einsum("ijk,kl->ijl",u_Q,w_u_Q)
                PP=tf.einsum("ijk,kl->ijl",u_P_t,w_u_P)
                PP_V=tf.einsum("ijk,kl->ijl",v_t_,w_v_P)
                logit=tf.tanh(QQ+PP+PP_V)
                logit=tf.einsum("ijk,kl->ijl",logit,V)
                soft_logit=tf.nn.softmax(logit,1)

                c_t=tf.einsum("ijk,ijl->ikl",soft_logit,u_Q)#(None,2*hideen_dim,1)
                c_t=tf.reshape(c_t,[-1,2*self.hidden_dim])#[None,2*hidden_dim]

                return c_t

    def self_match_attention_ops(self,P_,P_t):
        '''
        P_[None,P_len,hidden_dim],P_t[None,1,hidden_dim]
        :param P_: 
        :param P_t: 
        :return: 
        '''
        with tf.device('/gpu:0'):
            with tf.variable_scope("self_attention", reuse=True):
                w_u_P = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
                w_u_P_ = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
                V = tf.Variable(tf.random_normal(shape=(self.hidden_dim, 1)))

                PP = tf.einsum("ijk,kl->ijl", P_, w_u_P)
                PP_ = tf.einsum("ijk,kl->ijl", P_t, w_u_P_)
                logit = tf.tanh(PP+PP_)
                logit = tf.einsum("ijk,kl->ijl", logit, V)
                soft_logit = tf.nn.softmax(logit, 1)

                c_t = tf.einsum("ijk,ijl->ikl", soft_logit, P_)  # (None,hideen_dim,1)
                c_t = tf.reshape(c_t, [-1, self.hidden_dim])  # [None,hidden_dim]
                return c_t

    def self_match_attention(self,P_):
        '''
        自匹配注意力机制 
        :param P_: 
        :return: 
        '''
        P_P = tf.transpose(P_, [1, 0, 2])
        P_list = tf.unstack(P_P, self.P_len, 0)
        init_h = tf.zeros(shape=(self.batch_size, self.hidden_dim))
        lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                            state_is_tuple=True)
        H = [init_h]
        init_state=(init_h,init_h)
        state=[init_state]
        w_g = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, 2 * self.hidden_dim)))
        with tf.variable_scope("self_attention"):
            for t in range(self.P_len):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                P_t = tf.reshape(P_list[t], [-1, 1 , self.hidden_dim])

                c_t = self.self_match_attention_ops(P_,P_t)  # [None,hidden_dim]
                u_p_t_c_t = tf.concat((P_list[t], c_t), 1)
                u_p_t_c_t_ = tf.sigmoid(tf.matmul(u_p_t_c_t, w_g)) * u_p_t_c_t
                h_t, new_state = lstm_cell(u_p_t_c_t_, state[-1])  # v_t=[None,self.hidden_dim]
                H.append(h_t)
                state.append(new_state)
            out = H[1::]
            out = tf.stack(out)
            out = tf.transpose(out, [1, 0, 2])  # [None,P_len,hidden_dim]
            return out

    def pre_Q_attention(self,Q_):
        '''
        answer的pointer netwrok的init_state
        :param Q_: 
        :return: 
        '''

        Q_list=tf.unstack(Q_,self.Q_len,1)
        w_Q_1 = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.hidden_dim)))
        r_Q_1=tf.matmul(Q_list[-1],w_Q_1)
        # r_Q_1=Q_list[-1]
        with tf.variable_scope("pre_init_Q"):
            V_r_Q =  tf.zeros_like(tf.reshape(Q_list[0],(-1,1,2*self.hidden_dim)))
            w_u_Q = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            w_v_Q = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            V = tf.Variable(tf.random_normal(shape=(self.hidden_dim, 1)))

            u_Q_Q = tf.einsum("ijk,kl->ijl", Q_, w_u_Q)  # None,Q_len,hidden_dim
            v_Q_Q = tf.einsum("ijk,kl->ijl", V_r_Q, w_v_Q) #None,1,hidden_dim
            logit = tf.tanh(u_Q_Q + v_Q_Q) #None,Q_len,hidden_dim
            logit = tf.einsum("ijk,kl->ijl", logit, V)
            soft_logit = tf.nn.softmax(logit, 1)

            r_Q = tf.einsum("ijk,ijl->ikl", soft_logit, Q_)  # (None,2*hideen_dim,1)
            r_Q = tf.reshape(r_Q, [-1, 2*self.hidden_dim])  # [None,2*hidden_dim]
            w_Q = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            r_Q=tf.matmul(r_Q,w_Q)
            return r_Q,r_Q_1

    def pointer_attention(self,H,h_a_t_):
        '''
        H passage的输入 None,len_P,hidden_dim  h_a_t:None,hidden
        :param H: 
        :param h_a_t_: 
        :return: 
        '''
        with tf.device('/gpu:0'):
            H_list=tf.unstack(H,self.P_len,1)
            h_a_t_=tf.reshape(h_a_t_,(-1,1,self.hidden_dim))
            self.HHH=h_a_t_
            with tf.variable_scope("pointer_net_attention"):
                w_h_p = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
                w_a_p = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
                V = tf.Variable(tf.random_normal(shape=(self.hidden_dim, 1)))

                h_P_P=tf.einsum("ijk,kl->ijl", H, w_h_p)  # None,P_len,hidden_dim
                h_a_a=tf.einsum("ijk,kl->ijl", h_a_t_, w_a_p)  # None,1,hidden_dim
                logit = tf.tanh(h_P_P + h_a_a) #None,P_len,hidden_dim
                logit = tf.einsum("ijk,kl->ijl", logit, V)
                soft_logit = tf.nn.softmax(logit, 1) # None,P_len,1
                soft_logit_ = tf.reshape(soft_logit,(-1,self.P_len))
                id=tf.argmax(soft_logit_,1) # (None,)
                id_flatten=tf.reshape(id,(-1,1))
                H_flatten=tf.reshape(H,(-1,self.hidden_dim))
                ss=tf.gather(H_flatten,id_flatten)
                # print(ss)
                c_t=tf.einsum("ijk,ijl->ilk",soft_logit,H)#[None,hidden_dim,1]
                c_t=tf.reshape(c_t,(-1,self.hidden_dim))
                #c_t=tf.reshape(ss,(-1,self.hidden_dim))
                return soft_logit_,id,c_t

    def pointer_network(self,Q_,H):
        '''
        指针网络 pointer_network
        :return: 
        '''
        with tf.variable_scope("pointer_net"):
            init_c_state,init_input=self.pre_Q_attention(Q_) # [None,hidden_dim] [None,2*hidden_dim]
            # init_c_state = tf.transpose(H, [1, 0, 2])[-1]
            # cell=tf.contrib.rnn.LSTMCell(self.hidden_dim,
            #                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
            #                             state_is_tuple=True)

            cell=tf.contrib.rnn.BasicRNNCell(self.hidden_dim)
            #init_input=tf.zeros(shape=(self.batch_size,self.hidden_dim))
            #init_input=tf.transpose(H,[1,0,2])[-1]

            # init_input = tf.transpose(Q_, [1, 0, 2])[-1]
            # init_h_state=init_input
            # init_state=(init_c_state,init_h_state)
            # state=[init_state]
            state=[init_c_state]
            H_a=[init_input]
            soft_logits=[]
            ids=[]

            for t in range(self.out_num):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                soft_logit,id,c_t=self.pointer_attention(H,H_a[-1])#soft_logit [None,P_len] id [None]
                (h_t,new_state)=cell(c_t,state[-1])
                state.append(new_state)
                H_a.append(c_t)
                soft_logits.append(soft_logit)
                ids.append(id)
            soft_logits=tf.stack(soft_logits,1) #[None,pre_num,P_len]
            ids=tf.stack(ids,1)   #[batch_size,pre_num]
            return soft_logits,ids

    def __accuracy(self,predictions, labels):
        predict = np.argmax(predictions, 2)
        length = sum(1 for i, j in zip(predict.flatten(), labels.flatten()) if int(i) != 0 or int(j) != 0)
        ss = sum([1 for i, j in zip(predict.flatten(), labels.flatten()) if int(i) == int(j) and int(i) != 0])
        if float(length) == 0.0:
            return 0.0
        else:
            return 100.0 * (float(ss) / float(length))

    def train(self,dd):


        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                allow_soft_placement=True,
                                log_device_placement=True)
        dev_Q,dev_P,dev_Q_len,dev_P_len,dev_label=dd.get_dev(0,self.batch_size)
        # test_Q,test_A,test_label=dd.get_test()
        saver = tf.train.Saver()
        summary_write=tf.summary.FileWriter(FLAGS.summary_write_dir)
        with tf.Session(config=config) as sess:
            if os.path.exists(FLAGS.model_dir+'.index'):
                _logger.info("load model!")
                # sess.run(tf.global_variables_initializer())
                saver.restore(sess,FLAGS.model_dir)
            else:
                # sess.run(tf.global_variables_initializer())
                saver.restore(sess,FLAGS.model_dir)

            ini_acc=0.0
            init_loss=99.99
            best_index= 0
            for i in range(FLAGS.epoch):
                begin_time=time.time()
                Q,P,label,Q_len,P_len=dd.next_batch()
                Q_seq_vec=np.array(Q_len)
                A_seq_vec=np.array(P_len)
                train_loss,_,ids,merge_summary=sess.run([self.loss,self.optimizer,self.ids,self.merge_summary],
                                                        feed_dict={
                                                        self.Q:Q,
                                                          self.P:P,
                                                          self.Q_seq_vec:Q_seq_vec,
                                                          self.P_seq_vec:A_seq_vec,
                                                          self.label:label
                                              })
                summary_write.add_summary(merge_summary, i)
                if i%1==0:
                    _logger.info("迭代次数%s"%i)
                    _logger.info("训练误差：%s"%train_loss)
                    # _logger.info("训练误差：%s ,START_END_ID:%s"%(train_loss,ids))
                    _logger.info('best_index:%s||%s'%(best_index,init_loss))
                    end_time = time.time()
                    _logger.info("time:%s" % (end_time - begin_time))
                if train_loss<init_loss:
                    init_loss=train_loss
                    best_index=i
                    _logger.info("save %s"%i)
                    saver.save(sess,FLAGS.model_dir)


                # dev_softmax_out, dev_loss = sess.run([self.ids, self.loss],
                #                                  feed_dict={self.Q: dev_Q,
                #                                             self.P: dev_P,
                #                                             self.Q_seq_vec:dev_Q_len,
                #                                             self.P_seq_vec:dev_P_len,
                #                                             self.label: dev_label})
                # _logger.info("dev loss: %s"%dev_loss)

                _logger.info("*"*100)



    def infer(self,Q_array,P_array,Q_len,P_len):
        '''
        infer for Q_array ,P_array
        :return: 
        '''
        init_Q=np.zeros(self.Q_len)
        init_P=np.zeros(self.P_len)
        init_Q_len=self.Q_len
        init_P_len=self.P_len
        size=np.array(Q_array).shape[0]
        if np.array(Q_array).shape[0] <=self.batch_size:
            new_Q=list(Q_array)
            new_Q.extend([init_Q]*(self.batch_size-size))
            new_P=list(P_array)
            new_P.extend([init_P]*(self.batch_size-size))

            new_Q_len=list(Q_len)
            new_Q_len.extend([init_Q_len]*(self.batch_size-size))
            new_P_len=list(P_len)
            new_P_len.extend([init_P_len]*(self.batch_size-size))

        else:
            pass

        new_Q=np.array(new_Q)
        new_P=np.array(new_P)
        new_Q_len=np.array(new_Q_len)
        new_P_len=np.array(new_P_len)

        print(new_Q.shape)
        print(new_P.shape)
        print(new_Q_len.shape)
        print(new_P_len.shape)
        saver = tf.train.Saver()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                allow_soft_placement=True,
                                log_device_placement=True)
        with tf.Session(config=config) as sess:
            saver.restore(sess,FLAGS.model_dir)
            ids = sess.run(self.ids,
                 feed_dict={
                     self.Q: new_Q,
                     self.P: new_P,
                     self.Q_seq_vec: new_Q_len,
                     self.P_seq_vec: new_P_len})
            return ids[:size]

def main(_):

    dd = R_Net_Data(train_path=FLAGS.train_dir, test_path=FLAGS.test_dir,
                                dev_path=FLAGS.dev_dir, batch_size=FLAGS.batch_size,
                                Q_len=FLAGS.Q_len, P_len=FLAGS.P_len, flag="test")
    vocab_size=len(dd.vocab)
    ca = R_Net(embedding_dim=FLAGS.embedding_dim, hidden_dim=FLAGS.hidden_dim,
               Q_len=FLAGS.Q_len, P_len=FLAGS.P_len, batch_size=FLAGS.batch_size,vocab_size=vocab_size)
    _logger.info("This mod is %s"%FLAGS.mod)
    if FLAGS.mod=='train':
        for _ in range(20):
            ca.train(dd)
    elif FLAGS.mod=="infer":
        Q, P, label, Q_len, P_len, file_size=dd.get_infer_info(FLAGS.test_dir)
        ids=ca.infer(Q,P,Q_len,P_len)
        print(ids)
        print("*"*100)
        print(label)


if __name__ == '__main__':
    tf.app.run()


