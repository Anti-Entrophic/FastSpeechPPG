import logging

def logging_init():
  logging.basicConfig(level=logging.DEBUG #设置日志输出格式
      ,filename="/content/experiment.log" #log日志输出的文件位置和文件名
      ,format="%(asctime)s-%(levelname)s: %(message)s" #日志输出的格式
                      # -8表示占位符，让输出左对齐，输出长度都为8位
      ,datefmt="%Y-%m-%d %H:%M:%S"  #时间输出的格式
      ,force=True)

def log(message):
  logging.debug(message)