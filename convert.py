import os
import xml.etree.ElementTree as ET

xml_dir = "My Dataset/annotations"
img_dir = "My Dataset/images"

classes = {
    "minor_pothole": 0,
    "medium_pothole": 1,
    "major_pothole": 2
}

for xml_file in os.listdir(xml_dir):

    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    txt_file = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(img_dir, txt_file)

    with open(txt_path, "w") as f:

        for obj in root.findall("object"):

            cls_name = obj.find("name").text

            if cls_name not in classes:
                continue

            cls_id = classes[cls_name]

            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            f.write(
                f"{cls_id} {x_center} {y_center} {box_width} {box_height}\n"
            )

print("Conversion Completed")