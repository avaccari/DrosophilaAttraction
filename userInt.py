# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:06:18 2016
Name:    userInt.py
Purpose: Provide a basic user interface
Author:  Andrea Vaccari (av9g@virginia.edu)
Version: 0.0.0-alpha

    Copyright (C) Sat Oct  1 12:06:18 2016  Andrea Vaccari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import Tkinter as tk
import tkMessageBox as tkmb
import tkFileDialog as tkfd

class userInt(object):
    def __init__(self):
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.iconify()

    def chooseFile(self):
        fil = tkfd.askopenfilename()
        return fil

    def showInfo(self, txt):
        return tkmb.showinfo(title='INFO!',
                             message=txt,
                             icon=tkmb.INFO)

    def yesNo(self, txt):
        return tkmb.askyesno(title='YES/NO?',
                             message=txt,
                             icon=tkmb.QUESTION)
