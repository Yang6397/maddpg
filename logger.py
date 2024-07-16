# encoding: utf-8
# desc: 日志工具
import time

class logger:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(self.file_path, "w") as f:
            f.write("Log created at {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    def log(self, content, show=True, end='\n'):
        with open(self.file_path, "a") as f:
            f.write("{}: ".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("{}".format(content))
            f.write(end)
        if show:
            print(content, end=end)

if __name__ == "__main__":
    mylogger = logger("./test.log")
    mylogger.log("hello world!")
