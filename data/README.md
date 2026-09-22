# Dataset



## NASA/PROMISE KC1



This project uses the **KC1 software defect prediction dataset**, associated with the NASA Metrics Data Program and distributed through the PROMISE/tera-PROMISE ecosystem.



### Dataset identity



- **Dataset:** KC1

- **OpenML dataset ID:** 1067

- **OpenML version:** 1

- **Task:** Binary software defect prediction

- **Instances:** 2,109

- **Software metrics:** 21

- **Target variable:** `defects`

- **Non-defective instances:** 1,783

- **Defective instances:** 326



The dataset contains static software metrics such as lines of code, cyclomatic complexity, Halstead-related measures, operators/operands, and branch counts. The `defects` column is the binary prediction target.



## Reproducible acquisition



The raw dataset is retrieved from OpenML using scikit-learn:



```python

from sklearn.datasets import fetch_openml



dataset = fetch_openml(

&#x20;   data_id=1067,

&#x20;   as_frame=True,

&#x20;   parser="auto"

)



df = dataset.frame
