# Dataset Format 

## Universal Sketch Perceptual Grouping (SPG)

SPG dataset stores in a `.ndjson` format ([Newline Delimited JSON](https://en.wikipedia.org/wiki/JSON_streaming#Newline-delimited_JSON)). Each file coresponds to certain mode: `train`, `valid`, `test`. Then file name pattern looks like `SPG_{mode}.ndjson`. 

### Keys

Each row in `.ndjson` represents data sample in json format:  
```json
{key1: value1, key2: value2, ...}
{key1: value1, key2: value2, ...}
{key1: value1, key2: value2, ...}
...
```
Where key names is with keys "key_id", "recog_label", "seg_label", "seg_label1", "stroke_mask", "points_offsets", "position_list", "stroke_num", "drawing", "sketch_stroke_num", "sketch_components_num".

- **key_id**: `int | str(int_value)`  
    Unique sample id.  
    P.S. str(int_value) means that can be string that contains integer. Example: "213" - OK, "A32B3" - Not OK. 

- **record_label**: `int`  
    Label of an object that is represented by strokes.

- **seg_label**: List[int], shape: (MAX_STROKE_NUM, )  
    Where MAX_STROKE_NUM - maximum possible count of strokes in one sketch.  

    Vector of int values where each value represents stroke label. If total count of strokes less than MAX_STROKE_NUM than remaining componets of vector must be filled with a number that corresponds to "empty" label.  
    Example: if MAX_STROKE_NUM = 8, but we have only 4 strokes with labels (2, 4, 4, 5) and label 0 corresponds to "empty", then vector should be `[2, 4, 4, 5, 0, 0, 0, 0]`.

- **seg_label1**: List[int], shape: (MAX_STROKE_NUM, )  
    Vector of int values that represents stroke label. But instead of previous key where each cell of vector corresponds to appropriate stroke, this vector contains unique values of used classes.  
    Using example from **seg_label** section, this vector will be `[2, 4, 5, 0, 0, 0, 0, 0]`  


- **stroke_mask**: List[int], shape: (MAX_STROKE_NUM + 1, ) 
    Vector of 0/1 where 0 in cells where stroke exists and +1 ending else 1.  
    Example on data from **seg_label** section:  
    [0, 0, 0, 0, 0, 1, 1, 1, 1]



