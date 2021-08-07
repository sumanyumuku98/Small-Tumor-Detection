.RECIPEPREFIX +=

PYTHON=python
ROOT=data/DDSM_Wider_Face
TRAINDATA=$(ROOT)/ddsm_bbox.txt
VALDATA=$(ROOT)/wider_face_split/wider_face_val_bbx_gt.txt
TESTDATA=$(ROOT)/wider_face_split/wider_face_test_filelist.txt
DEVICE=3

CHECKPOINT=weights/checkpoint_50.pt

main: 
        CUDA_VISIBLE_DEVICES=$(DEVICE) $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT)

resume: 
        CUDA_VISIBLE_DEVICES=$(DEVICE) $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --epochs $(EPOCH)

evaluate: 
        CUDA_VISIBLE_DEVICES=$(DEVICE) $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val

evaluation:
        cd eval_tools/ && octave wider_eval.m

test: 
        CUDA_VISIBLE_DEVICES=$(DEVICE) $(PYTHON) evaluate.py $(TESTDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split test

cluster: 
        cd utils; CUDA_VISIBLE_DEVICES=$(DEVICE) $(PYTHON) cluster.py $(TRAIN_INSTANCES)

debug: 
        CUDA_VISIBLE_DEVICES=$(DEVICE) $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --batch_size 1 --workers 0 --debug

debug-evaluate: 
        CUDA_VISIBLE_DEVICES=$(DEVICE) $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val --batch_size 1 --workers 0 --debug
