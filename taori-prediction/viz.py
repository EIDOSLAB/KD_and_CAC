from enum import Enum
import logging as log
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from typing import Tuple

class SliceDirection(Enum):
    AXIAL = 0
    CORONAL = 1
    SAGITTAL = 2


class VolumePlot:
    def __init__(self, img, figsize: Tuple[int,int]=(8, 8), vmin: float=None, vmax: float=None) -> None:
        log.getLogger('PIL').setLevel(log.ERROR)
        log.getLogger('matplotlib').setLevel(log.ERROR)
        self.img = img
        self.direction = SliceDirection.AXIAL
        self.figsize = figsize
        self.vmin = vmin
        self.vmax = vmax

    def _subplot(self, fig, rows, cols, position, title, img, show_axis=True):
        subplot = fig.add_subplot(rows, cols, position)
        if title is not None:
            subplot.set_title(title)
        if not show_axis:
            plt.xticks([])
            plt.yticks([])
        if self.vmin is not None and self.vmax is not None:
            plt.imshow(img, cmap='gray', vmin=self.vmin, vmax=self.vmax)
        else:
            plt.imshow(img, cmap='gray')

    def plot(self, point: list[int] = None) -> None:
        fig = plt.figure(figsize=self.figsize)

        if point is None: # plot 9 planes at regular positions
            idx_step = self.img.shape[0] / 4
            row_step = self.img.shape[1] / 4
            col_step = self.img.shape[2] / 4

            for i in range(0,3):
                idx = int(idx_step * (i+1))
                row = int(row_step * (i+1))
                col = int(col_step * (i+1))
                self._subplot(fig, 3, 3, (i*3) + 1, f'index {idx}',  self.img[idx,:,:])
                self._subplot(fig, 3, 3, (i*3) + 2, f'row {row}',    self.img[:,row,:])
                self._subplot(fig, 3, 3, (i*3) + 3, f'column {col}', self.img[:,:,col])
        else: # plot 3 planes passing for the point
            self._subplot(fig, 1, 3, 1, f'index {point[0]}',  self.img[point[0],:,:])
            self._subplot(fig, 1, 3, 2, f'row {point[1]}',    self.img[:,point[1],:])
            self._subplot(fig, 1, 3, 3, f'column {point[2]}', self.img[:,:,point[2]])

        plt.subplots_adjust(hspace=0.25)
        plt.show()
        
    def plot_boxed(self, z: Tuple[int,int], y: Tuple[int,int], x: Tuple[int,int],
                   center: Tuple[int,int,int]=None,
                   border: int=3, border_value: int=1000):
        fig = plt.figure(figsize=self.figsize)
        
        boxed = self.img.detach().clone()
        w = border
        boxed[z[0]-w:z[0]+w,y[0]:y[1],x[0]:x[1]] = border_value
        boxed[z[1]-w:z[1]+w,y[0]:y[1],x[0]:x[1]] = border_value
        boxed[z[0]:z[1],y[0]-w:y[0]+w,x[0]:x[1]] = border_value
        boxed[z[0]:z[1],y[1]-w:y[1]+w,x[0]:x[1]] = border_value
        boxed[z[0]:z[1],y[0]:y[1],x[0]-w:x[0]+w] = border_value
        boxed[z[0]:z[1],y[0]:y[1],x[1]-w:x[1]+w] = border_value
        
        if center is None:
            center = (
                z[0]+((z[1]-z[0])//2),
                y[0]+((y[1]-y[0])//2),
                x[0]+((x[1]-x[0])//2),
            )
        
        self._subplot(fig, 1, 3, 1, f'index {center[0]}',  boxed[center[0],:,:])
        self._subplot(fig, 1, 3, 2, f'row {center[1]}',    boxed[:,center[1],:])
        self._subplot(fig, 1, 3, 3, f'column {center[2]}', boxed[:,:,center[2]])
        plt.show()

    def plot_matrix(self, size=8):
        fig = plt.figure(figsize=self.figsize)
        
        matrix_size = size*size
        if self.direction == SliceDirection.AXIAL:
            step = self.img.shape[0] / matrix_size
        elif self.direction == SliceDirection.CORONAL:
            step = self.img.shape[1] / matrix_size
        elif self.direction == SliceDirection.SAGITTAL:
            step = self.img.shape[2] / matrix_size
        for i in range(0, matrix_size):
            if self.direction == SliceDirection.AXIAL:
                self._subplot(fig, size, size, i+1, int(i*step), self.img[int(i*step),:,:], False)
            elif self.direction == SliceDirection.CORONAL:
                self._subplot(fig, size, size, i+1, int(i*step), self.img[:,int(i*step),:], False)
            elif self.direction == SliceDirection.SAGITTAL:
                self._subplot(fig, size, size, i+1, int(i*step), self.img[:,:,int(i*step)], False)
        plt.tight_layout(pad=0.2)
        plt.show()
        
    def plot_interactive(self):
        fig = plt.figure(figsize=self.figsize)

        slider_ax = fig.add_axes([0.97, 0.15, 0.025, 0.73])
        self.slider = Slider(
            ax=slider_ax,
            label='',
            valmin=0,
            valmax=self.img.shape[self.direction.value],
            valstep=1,
            orientation="vertical"
        )

        def update(val):
            if self.direction == SliceDirection.AXIAL:
                self.slider.label = 'index'
                self.idx_slice = int(val)
                self.row_slice = slice(0, self.img.shape[1])
                self.col_slice = slice(0, self.img.shape[2])
            elif self.direction == SliceDirection.CORONAL:
                self.slider.label = 'row'
                self.idx_slice = slice(0, self.img.shape[0])
                self.row_slice = int(val)
                self.col_slice = slice(0, self.img.shape[2])
            elif self.direction == SliceDirection.SAGITTAL:
                self.slider.label = 'column'
                self.idx_slice = slice(0, self.img.shape[0])
                self.row_slice = slice(0, self.img.shape[1])
                self.col_slice = int(val)

            self.slider.valmax = self.img.shape[self.direction.value]
            self.slider.ax.set_ylim(self.slider.valmin, self.slider.valmax)

            plt.imshow(self.img[self.idx_slice, self.row_slice, self.col_slice], cmap='gray')
        self.slider.on_changed(update)

        plus_ax = fig.add_axes([0.965, 0.9, 0.03, 0.04])
        plus = Button(plus_ax, '+', hovercolor='0.975')
        def update_plus(_):
            self.slider.set_val(self.slider.val+1)
        plus.on_clicked(update_plus)

        minus_ax = fig.add_axes([0.965, 0.07, 0.03, 0.04])
        minus = Button(minus_ax, '-', hovercolor='0.975')
        def update_minus(_):
            self.slider.set_val(self.slider.val-1)
        minus.on_clicked(update_minus)

        radio_ax = fig.add_axes([0, 0.01, 0.1, 0.1])
        radio = RadioButtons(radio_ax, ('Axial', 'Coronal', 'Sagittal'))
        def update_radio(val):
            direction_dict = {'Axial': SliceDirection.AXIAL, 'Coronal': SliceDirection.CORONAL, 'Sagittal': SliceDirection.SAGITTAL}
            self.direction = direction_dict[val]
            self.slider.set_val(self.img.shape[self.direction.value] // 2)
        radio.on_clicked(update_radio)

        fig.add_subplot(1, 1, 1)
        self.slider.set_val(self.img.shape[self.direction.value] // 2)
        plt.show()
