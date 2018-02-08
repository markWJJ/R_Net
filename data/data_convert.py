'''
将SQuAD 数据转化为txt格式
'''
import json
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data_convert")

class data_convert(object):

    def __init__(self,train_dir,dev_dir):
        self.train_dir=train_dir
        self.dev_dir=dev_dir

    def process(self,read_dir,write_dir,num):
        train_file=json.load(open(read_dir,'rb'))
        data=train_file['data'] #

        write_file= open(write_dir,'w')
        # 解析json数据
        index=0
        dataList=[]
        for e in data:
            for ee in e['paragraphs']:
                qa_all=ee['qas']
                for eee in qa_all:
                    content = ee['context'].replace('\n',"")
                    ans_list=self.answer_convert(eee['answers'],content)

                    index += 1
                    # ss=[eee['question'],content,ans_list]
                    # dev_list.append(ss)
                    question=str(eee['question']).strip()

                    dataList.append([index,question,content,"-".join(ans_list)])
        for e in dataList[0:num]:
            write_file.write(e[1])
            write_file.write('\t\t')
            write_file.write(e[2])
            write_file.write('\t\t')
            write_file.write(e[3])
            write_file.write('\n')
        _logger.info('写入%s完毕'%write_dir)

    def answer_convert(self,answers,content):
        '''
        将list[dict]形式答案 转化为 [start_index,end_index]
        :param answers: 
        :return: 
        '''
        s_list=[]
        fin_list=[]
        for e in answers:
            ans_start=int(e['answer_start'])
            content_list=str(content[:ans_start]).split(" ")
            content_list=[e for e in content_list if e]
            start_id = len(content_list)
            text=e['text']
            text_len= len(str(text).split(" "))
            end_id=int(start_id)+text_len
            # print(content[:ans_start])
            # print(start_id,end_id)
            s_list.append(str(start_id)+"-"+str(end_id))
        s_list=list(set(s_list))
        s_list=[[e.split('-')[0],e.split('-')[1]] for e in s_list] #只取一个答案

        return s_list[0]





if __name__ == '__main__':
    num_list=[20,200,5000,20000]

    train_dir='./train-v1.1.json'
    dev_dir='./dev-v1.1.json'
    dev_write_dir='./dev_out.txt'
    train_write_dir_list=['./train_out_%s.txt'%str(i) for i in num_list]


    dc=data_convert('./train-v1.1.json','./dev-v1.1.json')
    dc.process(dev_dir,dev_write_dir,-1)
    for num,train_write_dir in zip(num_list,train_write_dir_list):
        dc.process(train_dir,train_write_dir,num)

