import torch
import torch.nn as nn



import torch
import torch.nn as nn


class NextStepClassifier(nn.Module):
    def __init__(self, in_features_shape, num_steps, num_step_classes,encoder,):
        super(NextStepClassifier, self).__init__()
        
        self.encoder=encoder
        dummy_input=torch.zeros(1,*in_features_shape)
        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        # print("encoder",self.encoder)
        # print("dummy_input",dummy_input.shape)
        self.in_features_size=self.encoder(dummy_input).numel()
        #self.recorded_steps=RestoredSteps(num_steps=num_steps,num_options=num_step_classes,)
       
        
        
        self.nets = nn.ModuleList(
            [nn.LSTMCell(input_size= self.in_features_size, hidden_size=num_step_classes*num_steps ) for _ in
             range(num_steps)])
        self.num_steps = num_steps
        self.num_step_classes = num_step_classes
        #print("init num step classes",num_step_classes)


    def forward(self, features, previous, task):
        # print("classifier vars",vars(self))
        # print("")
        # print("forward", self.nets, features)
        #print("Next Steps Features",features.shape)
        batch = features.size(0)
        #print("encoder",self.encoder)
        #print("features device",features.device)
        #print("features",features.shape)
        features=self.encoder(features)
        #print("Next Steps Features encode",features.shape)
        features=features.view(batch,-1)
        features_size=features[0].numel()
        
        if features_size!=self.in_features_size:
            features=nn.functional.pad(features,(0,0,0,self.in_features_size-features_size))
        
        #print("feature shape after",features.shape)
        hidden_long_term = task.hidden_long_term[self] if task.hidden_long_term.get(self) is not None else torch.zeros(
            batch, self.num_step_classes *self.num_steps, device=features.device)

        if previous is None:
            hx = torch.zeros(batch, self.num_step_classes ,self.num_steps,
                                            device=features.device)
        else:
            hx = previous #.clone().detach()
     
        if hx.size(1)!=self.num_step_classes or hx.size(2)!=self.num_steps:
            hx=hx.unsqueeze(0)
            hx=torch.nn.functional.interpolate(hx,(self.num_step_classes,self.num_steps ,))
            #print("interpolated",hx.shape)
            #hx=hx[0]
        hx=hx.view(batch,-1)
      
        cx = hidden_long_term
        #print("hx",hx.shape,"cx",cx.shape,"features",features.shape)
       
        #result=[]
        
        for net in self.nets:
            #print("forward2", net, features.shape, hx.shape, cx.shape)
            #print("forward2",features.shape,hx.shape,cx.shape)
            hx, cx = net(features, (hx, cx))
            #result.append(hx)
            #print("forward2 result", net, features.shape, hx.shape, cx.shape)
        #result=torch.cat(result,dim=1)
      
        hx=hx.view(batch, self.num_step_classes, self.num_steps)
        task.hidden_long_term[self] = cx.detach()
        return hx
