import json
import re
import matplotlib.path as mpath
import numpy as np

category_dict = {0: 'title',              # 标题
 1: 'plain text',         # 文本
 2: 'abandon',            # 包括页眉页脚页码和页面注释
 3: 'figure',             # 图片
 4: 'figure_caption',     # 图片描述
 5: 'table',              # 表格
 6: 'table_caption',      # 表格描述
 7: 'table_footnote',     # 表格注释
 8: 'isolate_formula',    # 行间公式（这个是layout的行间公式，优先级低于14）
 9: 'formula_caption',    # 行间公式的标号

 13: 'inline_formula',    # 行内公式
 14: 'isolated_formula',  # 行间公式
 15: 'ocr_text'}              # ocr识别结果


def remove_latex_commands(latex_content):
    # 移除所有 LaTeX 命令（形如 \command{...} 或 \command[...] 或 \command...）
    clean_content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', latex_content)
    clean_content = re.sub(r'\\[a-zA-Z]+\[[^\]]*\]', '', clean_content)
    clean_content = re.sub(r'\\[a-zA-Z]+\b', '', clean_content)
    
    # 移除 LaTeX 环境（形如 \begin{environment} ... \end{environment}）
    clean_content = re.sub(r'\\begin\{[^}]*\}[^\\]*\\end\{[^}]*\}', '', clean_content, flags=re.DOTALL)

    # 移除注释（形如 % ...）
    clean_content = re.sub(r'^%.*', '', clean_content)
    
    # 移除^
    clean_content = re.sub(r'^\^', '', clean_content)
    
    # 移除多余的空白行
    clean_content = re.sub(r'\n\s*\n', '\n', clean_content)
    
    # 使用正则表达式匹配类似于 `+4.7\%` 的格式
    pattern = re.compile(r'([+-]?)(\d+(\.\d+)?)\\%')
    
    # 将匹配的字符串替换为去掉加号，但保留百分号的格式
    normalized_string = pattern.sub(lambda m: f"{m.group(2)}%", clean_content)
    
    return normalized_string

def point_to_line_distance(point, line_start, line_end):
    """
    计算点到线段的最短距离
    """
    # 向量化计算
    line = np.array(line_end) - np.array(line_start)
    point_to_start = np.array(point) - np.array(line_start)
    line_length = np.dot(line, line)
    if line_length == 0:
        return np.linalg.norm(point_to_start)
    projection = np.dot(point_to_start, line) / line_length
    projection = max(0, min(1, projection))
    closest_point = np.array(line_start) + projection * line
    return np.linalg.norm(point - closest_point)

def is_content_inside_box(box_points, content_points, tolerance=0):
    """
    判断内容是否在框框里，允许一定的距离误差
    
    :param box_points: 框框的四个顶点的坐标，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param content_points: 内容的四个顶点的坐标，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param tolerance: 容许误差（阈值），如果点到边界的最短距离小于等于该值，则认为点在框框内
    :return: 内容是否在框框内，True 表示在内，False 表示不在
    """
    # 创建表示框框的路径对象
    box_path = mpath.Path(box_points)
    
    # 判断内容的四个点是否都在框框内或在容许误差范围内
    for point in content_points:
        if not box_path.contains_point(point):
            # 如果点不在框框内，检查其是否在容许误差范围内
            is_near = False
            for i in range(len(box_points)):
                line_start = box_points[i]
                line_end = box_points[(i + 1) % len(box_points)]
                distance = point_to_line_distance(point, line_start, line_end)
                if distance <= tolerance:
                    is_near = True
                    break
            if not is_near:
                return False
    
    return True

def sort_elements_by_y(elements):
    def sort_key(element):
        center = calculate_center(transfer_poly(element["poly"]))
        center_y = center[1]
        center_x = center[0]
        return (round(center_y / 20), center_x)

    # 使用sorted，并在key中加入次要排序条件
    elements = sorted(elements, key=sort_key)
    return elements

def get_content_inside_box(box_points, elements):
    elements = sort_elements_by_y(elements)
    text_list = []
    # 判断内容的四个点是否都在框框内
    for element in elements:
        if is_content_inside_box(box_points,transfer_poly(element["poly"]),tolerance=5):
            try :
                if "text" in element.keys():
                    text_list.append(element["text"])
                if "latex" in element.keys():
                    text_list.append(remove_latex_commands(element["latex"]))
            except:
                pass
    return text_list

def transfer_poly(ploy:list):
    x1, y1, x2, y2,x3, y3, x4, y4 = ploy
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def get_element_of_category(elements,category_id):
    res =[]
    for item in elements:
        if item["category_id"] == category_id:
            res.append(item)
    return res

def calculate_center(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    return (center_x, center_y)

if __name__ == "__main__":
    with open("/home/songze/PDF-Extract-Kit/output/宁德时代：1H24业绩符合预期，盈利能力持续强劲.json",
            "r",
            encoding="utf-8") as file:
        result = json.load(file)
    all_text_list = []
    for page_result_inex,page_result in enumerate(result):
        all_text_list.append(f"page:{page_result_inex}")
        print(f"page:{page_result_inex}")
        elements_dict = {}
        for key_id in category_dict.keys():
            elements_dict[category_dict[key_id]] = sort_elements_by_y(get_element_of_category(page_result["layout_dets"],key_id))
        for element in sort_elements_by_y(elements_dict["plain text"]):
            text_list = get_content_inside_box(transfer_poly(element["poly"]),elements=page_result["layout_dets"])
            if text_list == []:
                continue
            if element in elements_dict["plain text"]:
                all_text_list.append("plain text:")
                print("plain text:")
            elif element in elements_dict["title"]:
                all_text_list.append("title:")
                print("title:")
            for text in text_list:
                all_text_list.append(text)
                print(text)
        all_text_list.append("")
        print("")

    # 打开一个文件，模式为 'w'，表示写入（会覆盖文件）
    with open("/home/songze/PDF-Extract-Kit/songze_code/output.txt", "w") as file:
        # 遍历列表，并逐行写入文件
        for line in all_text_list:
            file.write(line + "\n")  # 在每行后面加上换行符