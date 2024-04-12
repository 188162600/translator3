class NextStepsForEncoderAttn:
    def __init__(self,instance) -> None:
        self.instance=instance
    def get_for_q_proj(self):
        return self.instance.get_for_encoder_attn_q_proj()
    def get_for_k_proj(self):
        return self.instance.get_for_encoder_attn_k_proj()
    def get_for_v_proj(self):
        return self.instance.get_for_encoder_attn_v_proj()
    def get_for_out_proj(self):
        return self.instance.get_for_encoder_attn_out_proj()
class NextSteps:
    def __init__(self,logits,cfg,encoder_decoder_cfg=None,index=0) -> None:
        self.logits=logits
        self.index=index
        self.cfg=cfg
        self.encoder_decoder_cfg=encoder_decoder_cfg
    def requires_grad_for_layer(self,index):
        return True
    def get_for_encoder_attn(self):
        return NextStepsForEncoderAttn(self)
    def get_layers(self):
        return self.encoder_decoder_cfg.transformer_layers
    def get_for_layer(self,index):
        return NextSteps(self.logits,self.cfg,self.encoder_decoder_cfg,index+self.index,)
    def get_for_encoder(self):
        return NextSteps(self.logits,self.cfg,self.cfg.encoder,0)
    def get_for_decoder(self):
        return NextSteps(self.logits,self.cfg,self.cfg.decoder,self.cfg.encoder.selective_layers)
    def get_for_fc1(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.fc1_selection_index]
    def get_for_fc2(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.fc2_selection_index]
    def get_for_q_proj(self):
        # print("self.logits.next",self.logits.shape)
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_q_proj_selection_index]
    def get_for_k_proj(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_k_proj_selection_index]
    def get_for_v_proj(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_v_proj_selection_index]
    def get_for_out_proj(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.self_attn_out_proj_selection_index]
    def get_for_encoder_attn_q_proj(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_q_proj_selection_index]
    def get_for_encoder_attn_k_proj(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_k_proj_selection_index]
    def get_for_encoder_attn_v_proj(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_v_proj_selection_index]
    def get_for_encoder_attn_out_proj(self):
        return self.logits[:,self.index,self.encoder_decoder_cfg.encoder_attn_out_proj_selection_index]
        