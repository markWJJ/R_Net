'''
将SQuAD 数据转化为txt格式
'''
import json
class data_convert(object):

    def __init__(self,train_dir,dev_dir,write_train_dir,write_dev_dir):
        self.train_dir=train_dir
        self.dev_dir=dev_dir
        # self.process(self.dev_dir,write_dev_dir)
        self.process(self.train_dir,write_train_dir)


    def process(self,read_dir,write_dir):
        train_file=json.load(open(read_dir,'rb'))
        data=train_file['data'] #
        # e=data[0]
        # ee=e['paragraphs'][0]
        # content=ee['context']
        # qa_all=ee['qas']
        # for eee in qa_all:
        #     # ans_list=self.answer_convert(eee['answers'])
        #     print(self.answer_convert(eee['answers'],content))
        #     print(content[int(eee['answers'][0]["answer_start"])])
        #     print("*"*10)
        #     question=str(eee['question']).strip()
        #     content=content.replace('\n',"")

        write_file= open('./train_out_2w.txt','w')
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
        #             write_file.write(str(index))
        #             write_file.write("\t\t")
        #             write_file.write(question)
        #             write_file.write("\t\t")
        #             write_file.write(content)
        #             write_file.write("\t\t")
        #             write_file.write("-".join(ans_list))
        #             write_file.write('\n')
        # write_file.close()
        for e in dataList[0:20000]:

            # write_file.write(str(e[0]))
            # write_file.write('\t\t')
            write_file.write(e[1])
            write_file.write('\t\t')
            write_file.write(e[2])
            write_file.write('\t\t')
            write_file.write(e[3])
            write_file.write('\n')
            # write_file.write()
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
        # for e in s_list:
        #     start_id=len(str(content[:int(e[0])-1]).split(" "))
        #     end_id=start_id+(int(e[1])-int(e[0]))
        #     fin_list.append([str(start_id),str(end_id)])
        return s_list[0]





if __name__ == '__main__':
    dc=data_convert('./train-v1.1.json','./dev-v1.1.json','./train_out_2w.txt','./dev_out.txt')

