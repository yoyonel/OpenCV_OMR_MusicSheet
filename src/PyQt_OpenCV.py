# -*- coding: utf-8 -*-


import numpy
import cv2
from PyQt4.QtGui import *
from PyQt4.QtCore import *


class MyDialog(QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)

        self.cvImage = cv2.imread(r'Page_09_Pattern_26.png')
        height, width, byteValue = self.cvImage.shape
        byteValue = byteValue * width

        cv2.cvtColor(self.cvImage, cv2.COLOR_BGR2RGB, self.cvImage)

        self.mQImage = QImage(self.cvImage, width, height, byteValue, QImage.Format_RGB888)

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self.mQImage)
        painter.end()

    def keyPressEvent(self, QKeyEvent):
        super(MyDialog, self).keyPressEvent(QKeyEvent)
        if 's' == QKeyEvent.text():
            cv2.imwrite("cat2.png", self.cvImage)
        else:
            app.exit(1)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MyDialog()
    w.resize(600, 400)
    w.show()
    app.exec_()
