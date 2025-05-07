python main.py --dataname TADPOLE --mode local-global --marl_mode 2  --MP_mode GCN  --n_head 3 --n_hidden 21 --GC_mode weighted-cosine \
         --reg 0.26 --lr 0.01 --dropout 0.005 \
         --nlayer 2 --gl_attn_layer 0 --gl_lin_layer 1 \
         --CL_weight 0.5 --CL_rate 0.2  --CL_mse 0.9 \
         --CL_mode random_mask_edge




