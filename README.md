# Most Influential Object in SCVQA
 Work done for 5th year thesis
 
 This project is written to work upon the model given by https://github.com/jialinwu17/self_critical_vqa
 
 Task1.ipynb includes commands for execution, understanding and testing aspects of SCVQA model, and codes for purpose of preprocessing. It has many unecessary lines of codes, so please exercise caution while going through it. Please refer it to generate and replicate necessary input files used in this project. 
 
 Please note that the dataset.py file written in original SCVQA model is slightly modified for this project. Please make the changes to run this code on top of SCVQA model.
 
Replication of some of the results obtained in this project requires knowledge of Photoshop. MIO can be removed from the image using Clone Stamp tool of Photoshop.

Contents of folders:

0img : for replacing features with uniform noise
counterfactual analysis: For implementing counterfactual (cf) analysis by replacing features of identified MIO by noise
image exp : for visualising bounding boxes of MIO features (with and without cf analysis). Code files other than read3, vis_top5, vis_top5cf can be omitted.
train subset eval: codes for conducting experiments on training set.. can be omitted
trying creating subset: for understanding certain parts of SCVQA code .. please use the dataset.py and corresponding eval codes of this folder rather than original SCVQA code
understanding att_gv: can be omitted

Rest of the folders mostly follow the codes from these above mentioned folders. 

