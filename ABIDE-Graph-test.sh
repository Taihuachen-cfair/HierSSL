#--------------------------------------------
# base params:
# python main.py --dataname ABIDE --marl_mode 2  --n_head 3 --n_hidden 22 --MP_mode GCN --mode local-global --GC_mode weighted-cosine \
#          --reg 0.05481717459739162 --lr 0.002000441567593296 --dropout 0.35653644007463636 \
#          --nlayer 2 --gl_attn_layer 2 --gl_lin_layer 0 \
#          --CL_weight 0.6513080064919565 --CL_rate 0.5496061959919021 --CL_mse 0.6078815037069377 \
#          --CL_mode random_mask_edge --seed 54232  \
#           acc:91.39
#--------------------------------------------

python main.py --dataname ABIDE --marl_mode 2  --n_head 3 --n_hidden 22 --MP_mode GCN --mode local-global --GC_mode weighted-cosine \
          --reg 0.05481717459739162 --lr 0.002000441567593296 --dropout 0.35653644007463636 \
          --nlayer 2 --gl_attn_layer 2 --gl_lin_layer 0 \
          --CL_weight 0.6513080064919565 --CL_rate 0.5496061959919021 --CL_mse 0.6078815037069377 \
          --CL_mode random_mask_edge --seed 54232 \


