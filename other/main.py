import schedule
import time
import requests
from guorn.client import GuornClient
from Rrjj import Rrjj

baseUrl = "xx"
fundId = xx
authCode = 'xx'
client_jj = Rrjj(baseUrl, fundId, authCode)

client_gr = GuornClient(username='xx', password='xx', sid='xx', account_id='xx')
client_gr.login()


class Worker(object):

    def __init__(self, client_jj, client_gr):
        self.client_jj = client_jj
        self.client_gr = client_gr
        self.postion_jj = None
        self.postion_gr = None
        self.accInfo_jj = None

    def sync(self, ):
        # first 
        gr_client = self.client_gr
        jj_client = self.client_jj
        
        # strategy = gr_client.user_strategy()
        # print(strategy)

        acc_info = self.accInfo = jj_client.accInfo()
        print(acc_info)
        self.client_gr.auto_trader_instruction()

        # get position of gr

        # get position of jj



    # def get_acc_info(self, ):
    #     self.accInfo = self.client.accInfo()

    # def print_info(self, ):
    #     self.get_acc_info()
    #     print(self.accInfo)

worker1 = Worker(client_jj, client_gr)


schedule.every().seconds.do(worker1.sync)
# schedule.every().

if __name__ == "__main__":
    
    while True:
        schedule.run_pending()
        time.sleep(1)
