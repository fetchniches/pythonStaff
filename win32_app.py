from win32 import win32gui
import ctypes
from ctypes import *
from ctypes.wintypes import *
from pprint import pprint
 
def gbk2utf8(s):
    # return s.decode('gbk').encode('utf-8')
    return s

def show_window_attr(hWnd):
    '''
    显示窗口的属性
    :return:
    '''
    if not hWnd:
        return
 
    #中文系统默认title是gb2312的编码
    title = win32gui.GetWindowText(hWnd)
    title = gbk2utf8(title)
    clsname = win32gui.GetClassName(hWnd)
 
    print ('窗口句柄:{}'.format((hWnd)))
    print ('窗口标题:{}'.format(title)) 
    print ('窗口类名:{}'.format(clsname))
    print('-'*20)
 
def show_windows(hWndList):
    for h in hWndList:
        show_window_attr(h)
 
def demo_top_windows():
    '''
    演示如何列出所有的顶级窗口
    :return:
    '''
    hWndList = []
    win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), hWndList)
    show_windows(hWndList)
 
    return hWndList
 
def demo_child_windows(parent):
    '''
    演示如何列出所有的子窗口
    :return:
    '''
    if not parent:
        return
 
    hWndChildList = []
    win32gui.EnumChildWindows(parent, lambda hWnd, param: param.append(hWnd),  hWndChildList)
    show_windows(hWndChildList)
    return hWndChildList
 
def get_windows_coordinates(hwnd_name: str):
    "获取窗口坐标"
    hWnd = win32gui.FindWindow(None, hwnd_name)
    #获取句柄
    assert hWnd != 0
    #未找到句柄时引发异常 
    try:
        f = ctypes.windll.dwmapi.DwmGetWindowAttribute
    except WindowsError:
        f = None
    if f:
        rect = ctypes.wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        f(ctypes.wintypes.HWND(hWnd),
        ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
        ctypes.byref(rect),
        ctypes.sizeof(rect)
        )       

        return rect.left, rect.top, rect.right, rect.bottom

hWndList = demo_top_windows()
assert len(hWndList)
 
parent = hWndList[20]
#这里系统的窗口好像不能直接遍历，不知道是否是权限的问题
# hWndChildList = demo_child_windows(parent)
 
print('-----top windows-----')
# pprint(hWndList)
 
# print('-----sub windows:from %s------' % (parent))
# pprint(hWndChildList)
