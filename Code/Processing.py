import os
from openpyxl import Workbook

class Processing():
    def __init__(self):
        self.folder_name = '../CrossversionData'

    def write_excel(self, excel_path, data):
        '''
        :param path: Excel的路径
        :param data: 要写入的数据，是一个二维的list，每一个元素是一个一维的list
        :return:
        '''

        dir_name = str(os.path.split(excel_path)[0])  # 把目录分离出来，没有的话则创建
        print(dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        wb = Workbook()
        ws = wb.active
        for _ in data:
            ws.append(_)
        wb.save(excel_path)

