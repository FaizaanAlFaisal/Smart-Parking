import os
import numpy as np
import cv2
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector, TextBox
from matplotlib.collections import PatchCollection
from shapely.geometry import Point


class RegionSelector:
    """
    RegionSelector
    ------------
    This class is designed to be a compact solution for selecting and storing polygon regions in an image/video feed to use for any purpose.
    Specifically made for use alongside YOLO models, but can be easily modified to accommodate any purpose.

    The class utilizes matplotlib's Polygon Selector as a base.
    
    <br>

    Basic Tutorial for GUI usage:
        Select a region in the figure by enclosing them within a polygon.
        Press 'n' to make a new polygon.
        Press 'b' to save the currently drawn polygon.
        Press 'q' to exit the program and save the data.
        Hold 'Shift' while dragging to move entire shape.
        Hold 'Ctrl' while dragging to move a single vertex.
        Press 'Esc' to clear current drawn shape.

    <br>
    
    Parameters:
        img_path (str): Provide the relative path of image on which to define the regions in.
        img_dimensions (tuple(int, int)): Provide the dimensions (width, height) on which to resize image to/store coordinates for.
        
        
    """

    def __init__(self, img_path:str, img_dimensions:tuple[int, int]=(1280,720) ):
        self.img_path : str = img_path 
        self.img_dimensions : tuple[int, int] = img_dimensions

        self.current_polygon_selector = None
        self.selected_polygon = []
        self.stored_polygons = {}
        self.last_polygon_label = 0

        # create matplotlib base
        self.fig, self.ax= plt.subplots()

        # manage all button presses
        self.cid = self.ax.figure.canvas.mpl_connect('key_press_event', self.__on_key_press)
        
        # prepare background image for polygon selectors
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(rgb_img, img_dimensions)
        
        # setup display
        self.__draw_ax()
        plt.show()


    def __draw_ax(self):
        """
        Recreate the axes for the figure.
        """
        # fully clear the axes
        self.ax.clear()

        # reset the initialization of the axes
        self.ax.set_title("Press 'n' to start a new polygon. \nPress 'q' to save/exit.")
        self.ax.imshow(self.img)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # draw any polygons if there are any
        self.__draw_previous_polygons()


    def __on_select(self, verts):
        """
        Functionality to perform when a completed polygon has been selected (start point = end point).
        """
        self.selected_polygon = verts
        print(self.selected_polygon)

    
    def __on_key_press(self, event):
        """
        All functionality related to any button presses for the matplotlib window.
        """
        # make new polygon
        if event.key == "n" or event.key == "N":

            self.__start_new_polygon(self.ax)
        
        # save polygon
        if event.key == "b" or event.key == "B":
            if self.selected_polygon != []:
                self.__save_polygon()
    
    def __save_polygon(self):
        """
        Functionality after pressing the save button.
        """
        if self.current_polygon_selector is None:
            return

        # clear the polygon selector
        self.current_polygon_selector.clear()
        self.current_polygon_selector.disconnect_events()
        self.current_polygon_selector = None

        # save the selected polygon:
        self.stored_polygons[self.last_polygon_label] = self.selected_polygon
        self.last_polygon_label += 1
        self.selected_polygon = []

        self.__draw_ax()


    def __start_new_polygon(self, ax:Axes):
        """
        Functionality for starting a new polygon. If one already exists, clear it.
        """
        print("All stored polygons: ", self.stored_polygons)

        if self.current_polygon_selector is not None:
            self.current_polygon_selector.clear()
            self.current_polygon_selector.disconnect_events()
            self.current_polygon_selector = None
            
            # clear current polygon entry
            self.selected_polygon = []

        self.current_polygon_selector = PolygonSelector(ax, self.__on_select)
        self.ax.set_title("Press 'b' to save the polygon when polygon is completed.")


    def __draw_previous_polygons(self, color='navy', alpha=0.75):
        """
        Draw all polygons stored in the self.stored_polygons() var.
        """
        for polygons in self.stored_polygons.values():
            polygon = Polygon(polygons, closed=True, color=color, alpha=alpha, edgecolor='none')
            self.ax.add_patch(polygon)
            self.ax.figure.canvas.draw()


def main():
    obj = RegionSelector(img_path='data\imgs\parking_lot.jfif', img_dimensions=(474, 314))


if __name__ == '__main__':
    main()

