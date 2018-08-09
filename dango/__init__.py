# -*- coding: utf-8 -*-

'''
class DeepArchitecture(object):
    def __init__(self, **kwargs):
        return None

ConvNet = DeepArchitecture()
'''

from dango.dataio.image.records_visulizer import TFRecordVisulizer
from dango.utilities import parameters_parser
from dango.engine.dirver import Driver


def main():
    arguments = parameters_parser.parse()
    driver = Driver(arguments)
    driver.run()

    # checker = TFRecordVisulizer(driver._data_provider)
    # print('total number: ', checker.records_num)
    # checker.visualize()


