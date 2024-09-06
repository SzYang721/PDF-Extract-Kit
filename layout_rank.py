import json
import os

# Category dictionary
category_dict = {
    0: 'title',
    1: 'plain text',
    2: 'abandon',
    3: 'figure',
    4: 'figure_caption',
    5: 'table',
    6: 'table_caption',
    7: 'table_footnote',
    8: 'isolate_formula',
    9: 'formula_caption',
    10:'',
    11:'',
    12:'',
    13:'inline_formula',
    14:'isolated_formula',
    15:'ocr_text'
}


def sort_key(element):
        # Define a threshold for "closeness"
        threshold = 10  # Adjust based on your specific needs
        
        # Get the bounding box coordinates, corrected indexing to retrieve xmin, ymin, xmax (top-right x)
        xmin, ymin, _, _, xmax, ymax, _, _ = element['poly']

        # Return a tuple that includes a modified primary key that groups close elements together
        # and the secondary key for fine sorting within those groups
        # return (xmin // threshold, -ymin // threshold, xmax // threshold, -ymax // threshold) # left to right; top to bottom
        return (-ymax // threshold, xmin // threshold, -ymin // threshold, xmax // threshold)# top to bottom; left to right
   
 
def sort_elements(json_data):
    # Function to sort the elements based on the top-left and top-right corners

    # Process each page in the JSON
    for page in json_data:
        # Sort the layout detections on the current page
        page['layout_dets'] = sorted(page['layout_dets'], key=sort_key)
        
    return json_data


def sort_original_key(element):
        # Get the bounding box coordinates, corrected indexing to retrieve xmin, ymin, xmax (top-right x)
        xmin, ymin, _, _, xmax, ymax, _, _ = element['poly']

        # Return a tuple that includes a modified primary key that groups close elements together
        # and the secondary key for fine sorting within those groups
        # return (xmin, ymin, xmax, ymax) # left to right; top to bottom
        return (-ymax, xmin, -ymin, xmax) # top to bottom; left to right


def sort_by_primary_key(element):
        # Define a threshold for "closeness"
        threshold = 10  # Adjust based on your specific needs
        
        # Get the bounding box coordinates, corrected indexing to retrieve xmin, ymin, xmax (top-right x)
        xmin, ymin, _, _, xmax, ymax, _, _ = element['poly']

        # Return a tuple that includes a modified primary key that groups close elements together
        # and the secondary key for fine sorting within those groups
        return xmin


def sort_primary_key(element, index):
    # This key will sort primarily by xmin (left coordinate)
    return sort_original_key(element)[index]  # xmin


def sort_secondary_key(element, index):
    # Secondary sort by ymin (top coordinate)
    return sort_original_key(element)[index]  # ymin


def sort_by_moving_window(sorted_list, primary_key_index, secondary_key_index, threshold):
    """
    Refines sorting of a list sorted by a primary key using a secondary key within threshold groups.

    :param sorted_list: List of items sorted by the primary key.
    :param secondary_key: Function to extract the secondary sort key.
    :param primary_key_index: Index of the primary key in the item's tuple to compare.
    :param threshold: Threshold value to form groups for secondary sorting.
    :return: List sorted by the primary key and refined by the secondary key within each group.
    """
    if not sorted_list:
        return []

    # Initialize the refined result list and the first window with the first element
    refined_list = []
    window = [sorted_list[0]]

    # Iterate through each element in the sorted list starting from the second element
    for current_item in sorted_list[1:]:
        # Check if the current item is within the threshold range of the window's first item
        if abs(sort_original_key(current_item)[primary_key_index] - sort_original_key(window[0])[primary_key_index]) <= threshold:
            window.append(current_item)
        else:
            # If the window has more than one element, sort it by the secondary key
            if len(window) > 1:
                # print((sort_original_key(current_item)[secondary_key_index]))
                window.sort(key=lambda item: sort_secondary_key(item, secondary_key_index))
            # Extend the refined list with the sorted window and start a new window with the current item
            refined_list.extend(window)
            window = [current_item]

    # Sort and add the last window
    if len(window) > 1:
        window.sort(key=lambda item: sort_secondary_key(item, secondary_key_index))
    refined_list.extend(window)

    return refined_list

def sort_elements_moving_window(json_data, threshold):
    for page in json_data:
        # Sort the layout detections on the current page
        for i in range(4):
            # print(type(page['layout_dets']))
            if i == 0:
                page['layout_dets'] = sorted(page['layout_dets'], key=sort_by_primary_key)
            page['layout_dets'] = sort_by_moving_window(page['layout_dets'], i, i+1, threshold)
            if i+1 == 3:
                break
    return json_data
    

if __name__ == '__main__':
    layout_path = '/data/zl/pdf_extract/data/layout_text_json/债券研究/25286028.json'
    output_path = './output/'
    threshold = 20
    
    json_data = json.loads(open(layout_path).read())
    basename = os.path.basename(layout_path)[0:-4]

    # doc_layout_result = sort_elements(json_data)
    
    doc_layout_result = sort_elements_moving_window(json_data, threshold)
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'{basename}.json'), 'w') as f:
        json.dump(doc_layout_result, f)