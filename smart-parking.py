import os
import pickle


def main():
    file_path = "parking-spots.pkl"
    if not os.path.exists(file_path):
        print("No save detected. Please run select-regions.py to define polygon regions first.")
        exit()
        
    with open("parking-spots.pkl", 'rb') as file:
        stored_polygons = pickle.load(file)
        img_dimensions = pickle.load(file)

    print("Polygons:", stored_polygons)
    print("Image dimensions:", img_dimensions)


if __name__ == "__main__":
    main()