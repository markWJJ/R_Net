import numpy as np
import pickle
import os
global PATH
PATH=os.path.split(os.path.realpath(__file__))[0]
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("r_net_data")

class R_Net_Data(object):
    '''
    R_Net模型 数据处理模块
    '''
    def __init__(self,train_path,dev_path,test_path,batch_size,Q_len,P_len,flag):
        self.train_path=train_path  #训练文件路径
        self.dev_path=dev_path  #验证文件路径
        self.test_path=test_path    #测试文件路径
        self.Q_length=Q_len    # 问句长度
        self.P_length=P_len    # 文档长度
        self.batch_size=batch_size  #batch大小

        # self.label_dict={"0":0,"B":1,"M":2,"E":3,"S":4}
        if flag=="train_new":
            self.vocab=self.get_vocab()
            pickle.dump(self.vocab,open(PATH+"/vocab.p",'wb'))  # 词典
        elif flag=="test" or flag=="train":
            self.vocab=pickle.load(open(PATH+"/vocab.p",'rb'))  # 词典
        self.index=0
        self.Q, self.P, self.label, self.Q_len, self.P_len, self.file_size = self._train_data()
        if self.batch_size > self.file_size:
            _logger.error("batch规模大于训练数据规模！")

    def get_vocab(self):
        '''
        构造字典 dict{NONE:0,word1:1,word2:2...wordn:n} NONE为未登录词
        :return: 
        '''
        train_file=open(self.train_path,'r')
        test_file=open(self.dev_path,'r')
        dev_file=open(self.test_path,'r')
        vocab={"NONE":0}
        index=1
        for ele in train_file:
            ele.replace("\n","")
            ele1=ele.replace("\t\t"," ")
            ws=ele1.split(" ")
            for w in ws:
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1

        for ele in test_file:
            ele1=ele.replace("	"," ").replace("\n","")
            for w in ele1.split(" "):
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1

        for ele in dev_file:
            ele1=ele.replace("	"," ").replace("\n","")
            for w in ele1.split(" "):
                w=w.lower()
                if w not in vocab:
                    vocab[w]=index
                    index+=1

        train_file.close()
        dev_file.close()
        test_file.close()
        return vocab

    def sent2vec(self,sent,max_len):
        '''
        根据vocab将句子转换为向量
        :param sent: 
        :return: 
        '''

        sent=str(sent).replace("\n","")
        sent_list=[]
        real_len=len(sent.split(" "))
        for word in sent.split(" "):
            word=word.lower()
            if word in self.vocab:
                sent_list.append(self.vocab[word])
            else:
                sent_list.append(0)

        if len(sent_list)>=max_len:
            new_sent_list=sent_list[0:max_len]
        else:
            new_sent_list=sent_list
            ss=[0]*(max_len-len(sent_list))
            new_sent_list.extend(ss)
        sent_vec=np.array(new_sent_list)
        if real_len>=max_len:
            real_len=max_len
        return sent_vec,real_len

    def get_ev_ans(self,sentence):
        '''
        获取 envience and answer_label
        :param sentence: 
        :return: 
        '''
        env_list=[]
        ans_list=[]
        for e in sentence.split(" "):
            try:
                env_list.append(e.split("/")[0])
                ans_list.append(self.label_dict[str(e.split("/")[1])])
            except:
                pass
        return " ".join(env_list),ans_list

    def _train_data(self):
        '''
        获取训练数据
        :return: 
        '''
        train_file = open(self.train_path, 'r')
        Q_list = []  # 问句list
        P_list = []  # 文档list
        label_list = []  # label list
        file_size = 0
        Q_len_list = []
        P_len_list = []
        for sentence in train_file.readlines():
            file_size+=1
            sentence = sentence.replace("\n", "")
            sentences = sentence.split("\t\t")

            Q_sentence = sentences[0]   # 分词的问句 sentence
            P_sentence = sentences[1]   # 分词的文档 sentence
            label = [int(e) for e in sentences[2].split("-")]   # label
            Q_vec, Q_real_len = self.sent2vec(Q_sentence, self.Q_length)
            P_vec, P_real_len = self.sent2vec(P_sentence, self.P_length)

            Q_list.append(Q_vec)
            P_list.append(P_vec)

            Q_len_list.append(Q_real_len)
            P_len_list.append(P_real_len)

            label_list.append(label)
        train_file.close()
        result_Q = np.array(Q_list)
        result_P = np.array(P_list)
        result_Q_len_list = np.array(Q_len_list)
        result_P_len_list = np.array(P_len_list)

        result_label = np.array(label_list)

        Q, P, label,Q_len,P_len = self.shuffle_(result_Q, result_P, result_label,result_Q_len_list,result_P_len_list)[:]
        return Q, P, label,Q_len,P_len,file_size

    def shuffle_(self,*args):
        '''
        将矩阵X打乱
        :param x: 
        :return: 
        '''
        ss=list(range(args[0].shape[0]))
        np.random.shuffle(ss)
        new_res=[]
        for e in args:
            new_res.append(np.zeros_like(e))
        fin_res=[]
        for index,ele in enumerate(new_res):
            for i in range(args[0].shape[0]):
                ele[i]=args[index][ss[i]]
            fin_res.append(ele)
        return fin_res

    def next_batch(self):
        '''
        获取的下一个batch
        :return: 
        '''
        num_iter=int(self.file_size/self.batch_size)
        if self.index<num_iter:
            return_Q=self.Q[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_P=self.P[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_label=self.label[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_Q_len=self.Q_len[self.index*self.batch_size:(self.index+1)*self.batch_size]
            return_P_len=self.P_len[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
        else:
            self.index=0
            return_Q=self.Q[0:self.batch_size]
            return_P=self.P[0:self.batch_size]
            return_Q_len = self.Q_len[0:self.batch_size]
            return_P_len = self.P_len[0:self.batch_size]
            return_label=self.label[0:self.batch_size]
        return return_Q,return_P,return_label,return_Q_len,return_P_len

    def get_dev(self,begin_id,end_id):
        '''
        读取验证数据集
        :return: 
        '''
        dev_file = open(self.dev_path, 'r')
        Q_list = []
        P_list = []
        Q_len_list=[]
        P_len_list=[]
        label_list = []
        train_sentcens = dev_file.readlines()
        for sentence in train_sentcens:
            sentence = sentence.replace("\n", "")
            sentences = sentence.split("\t\t")
            Q_sentence=sentences[0]
            P_sentence=sentences[1]
            label = [int(e) for e in sentences[2].split("-")]   # label
            Q_array,Q_real_len=self.sent2vec(Q_sentence,self.Q_length)
            P_array,P_real_len=self.sent2vec(P_sentence,self.P_length)
            Q_len_list.append(Q_real_len)
            P_len_list.append(P_real_len)
            Q_list.append(list(Q_array))
            P_list.append(list(P_array))
            label_list.append(label)
        dev_file.close()
        result_Q=np.array(Q_list)
        result_P=np.array(P_list)
        result_Q_len=np.array(Q_len_list)
        result_P_len=np.array(P_len_list)
        result_label=np.array(label_list)
        return result_Q[begin_id:end_id],result_P[begin_id:end_id],result_Q_len[begin_id:end_id],result_P_len[begin_id:end_id],result_label[begin_id:end_id]

    def get_test(self):
        '''
        读取测试数据集
        :return: 
        '''
        test_file = open(self.test_path, 'r')
        Q_list = []
        A_list = []
        label_list = []
        train_sentcens = test_file.readlines()
        for sentence in train_sentcens:
            sentences=sentence.split("	")
            Q_sentence=sentences[0]
            A_sentence=sentences[1]
            label=sentences[2]
            Q_array,_=self.sent2array(Q_sentence,self.Q_len)
            A_array,_=self.sent2array(A_sentence,self.P_len)

            Q_list.append(list(Q_array))
            A_list.append(list(A_array))
            label_list.append(int(label))
        test_file.close()
        result_Q=np.array(Q_list)
        result_A=np.array(A_list)
        result_label=np.array(label_list)
        return result_Q,result_A,result_label

    def  get_Q_array(self,Q_sentence):
        '''
        根据输入问句构建Q矩阵
        :param Q_sentence: 
        :return: 
        '''
        Q_len=len(str(Q_sentence).replace("\n","").split(" "))
        if Q_len>=self.Q_len:
            Q_len=self.Q_len
        Q_array,_=self.sent2array(Q_sentence,self.Q_len)
        return Q_array,np.array([Q_len])

    def get_A_array(self,A_sentence):
        '''
        根据输入的答案句子构建A矩阵
        :param A_sentence: 
        :return: 
        '''
        A_sentence, label = self.get_ev_ans(A_sentence)
        P_len=len(label)
        if P_len>=self.P_len:
            P_len=self.P_len
        return self.sent2array(A_sentence,self.P_len)[0],np.array([P_len])

    def get_infer_info(self,infer_dir):
        '''
        
        :param infer_dir: 
        :return: 
        '''
        infer_data=open(infer_dir,'r')
        Q_list = []  # 问句list
        P_list = []  # 文档list
        label_list = []  # label list
        file_size = 0
        Q_len_list = []
        P_len_list = []
        for sentence in infer_data.readlines():
            file_size += 1
            sentence = sentence.replace("\n", "")
            sentences = sentence.split("\t\t")

            Q_sentence = sentences[0]  # 分词的问句 sentence
            P_sentence = sentences[1]  # 分词的文档 sentence
            label = [int(e) for e in sentences[2].split("-")]  # label
            Q_vec, Q_real_len = self.sent2vec(Q_sentence, self.Q_length)
            P_vec, P_real_len = self.sent2vec(P_sentence, self.P_length)

            Q_list.append(Q_vec)
            P_list.append(P_vec)

            Q_len_list.append(Q_real_len)
            P_len_list.append(P_real_len)

            label_list.append(label)
        infer_data.close()
        result_Q = np.array(Q_list)
        result_P = np.array(P_list)
        result_Q_len_list = np.array(Q_len_list)
        result_P_len_list = np.array(P_len_list)

        result_label = np.array(label_list)

        Q, P, label, Q_len, P_len = self.shuffle_(result_Q, result_P, result_label, result_Q_len_list,
                                                  result_P_len_list)[:]
        return Q, P, label, Q_len, P_len, file_size


if __name__ == '__main__':

    dd = R_Net_Data(train_path="./data/SQUQA_train_1.txt", test_path="./data/test.txt",
                            dev_path="./data/dev_out.txt", batch_size=4 ,Q_len=30, P_len=100, flag="train_new")
    # for i in range(100):
    #
    #     return_Q, return_P, return_label, return_Q_len, return_P_len = dd.next_batch()
    #     print(return_label)
    #     print("\n")
    Q,P,Q_,P_,L=dd.get_dev(0,10)
    print(Q)
