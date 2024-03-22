class NextSteps:
    def __init__(self, tensor: torch.Tensor,index:torch.Tensor=None,softmax:torch.Tensor=None,probability:torch.Tensor=None,confidence:torch.Tensor=None,restored_step_index=None):
        #print("tensor1",tensor.shape)
        
        self.tensor = tensor
        #print("tensor2",tensor.shape)
        
        if index is None:
            self.indices = torch.argmax(tensor,dim=1)
        else :
            self.indices=index
        # print(self.indices)
        # print("self.indices.shape,self.tensor.shape",self.indices.shape,self.tensor.shape)
        expanded_indices = self.indices.unsqueeze(-1)
        if softmax is None:
            self.softmax = torch.softmax(self.tensor, dim=1)
        else:
            self.softmax=softmax
       
        if probability is None:
            self.probability = torch.gather(self.softmax, 1,expanded_indices).squeeze(-1)
        else:
            self.probability=probability
        if confidence is None:
            #print("self.probability",self.probability.shape)
            self.confidence=torch.sum(self.probability,dim=1)
        else:
            self.confidence=confidence
        self.restored_step_index=restored_step_index
     

# class NextSteps:
#     def __init__(self, tensor: torch.Tensor):
#         #print("tensor1",tensor.shape)
        
#         self.tensor = tensor
#         #print("tensor2",tensor.shape)
        
       
#         self.indices = torch.argmax(tensor,dim=1)
       
#         # print(self.indices)
#         # print("self.indices.shape,self.tensor.shape",self.indices.shape,self.tensor.shape)
#         expanded_indices = self.indices.unsqueeze(-1)
       
#         self.softmax = torch.softmax(self.tensor, dim=1)
        
       
        
#         self.probability = torch.gather(self.softmax, 1,expanded_indices) .squeeze(-1)
#         #print("self.probability",self.probability)
    
#         self.confidence=torch.sum(self.probability,dim=1)
#         #print("self.confidence",self.confidence.shape)
        
class RestoredSteps:
    def __init__(self,num_steps,num_options,num_samples,num_old,num_new,num_fresh,old_new_fresh_distribution=None) -> None:
        self.softmax=torch.zeros(num_old+num_new+num_fresh,num_options,num_steps)
        self.indices=torch.zeros(num_old+num_new+num_fresh,num_steps,dtype=torch.long)
        self.losses=torch.zeros(num_old+num_new+num_fresh,num_samples)
        self.occurrences=torch.zeros(num_old+num_new+num_fresh,dtype=torch.long)
        self.old_new_fresh_distribution=old_new_fresh_distribution
        
        
        self.current_sample=torch.zeros(num_old+num_new+num_fresh,dtype=torch.long)
        self.num_samples=num_samples
        self.tracking_index=num_new+num_old
        
         
        torch.fill_(self.losses,math.inf)
       
        self.num_fresh=num_fresh
        self.num_old=num_old
        self.num_new=num_new
    def to(self,*args,**kwargs):
        self.softmax=self.softmax.to(*args,**kwargs)
        self.indices=self.indices.to(*args,**kwargs)
        self.losses=self.losses.to(*args,**kwargs)
        self.occurrences=self.occurrences.to(*args,**kwargs)
        self.current_sample=self.current_sample.to(*args,**kwargs)
        return self
    @torch.no_grad()
    def aggregate_losses(self):
        # Unique occurrences
        unique_occurrences, indices = torch.unique(self.occurrences, return_inverse=True)
        # Initialize tensor for aggregated losses
        aggregated_losses = torch.zeros_like(unique_occurrences, dtype=torch.float)

        for i, occ in enumerate(unique_occurrences):
            # Indices of all tracking events with the current occurrence
            idx = (indices == i)
            # Select the corresponding losses and sample counts
            occ_losses = self.losses[idx]
            occ_samples = self.current_sample[idx]

            # Aggregate losses by calculating the mean for each occurrence
            total_loss = 0
            total_samples = 0
            for j, samples in enumerate(occ_samples):
                if samples > 0:  # Ensure division is meaningful
                    total_loss += occ_losses[j, :samples].sum()
                    total_samples += samples
            if total_samples > 0:  # Avoid division by zero
                aggregated_losses[i] = total_loss / total_samples
            # else:
            #     aggregated_losses[i]=math.inf

        return unique_occurrences.float(), aggregated_losses
    @torch.no_grad()
    def linear_interp_loss(self, occurrence):
        # Assuming torch_linear_interp is defined as before
        occurrences, aggregated_losses = self.aggregate_losses()

        # Make sure occurrences are sorted
        sorted_indices = torch.argsort(occurrences)
        sorted_occurrences = occurrences[sorted_indices]
        sorted_losses = aggregated_losses[sorted_indices]

        # Interpolate
        interp_val = linear_interp(occurrence, sorted_occurrences, sorted_losses)
        return interp_val
    @torch.no_grad()
    def get_losses(self):
        sum_loss=self.losses.sum(dim=1)
        return sum_loss/self.current_sample
    @torch.no_grad()
    def get_efficiency(self):
        expected_loss=self.linear_interp_loss(self.occurrences)
        diff= expected_loss-self.get_losses()
        return diff
        
 
    @torch.no_grad()
    def track(self,loss,next_steps:NextSteps):
        #print("loss",loss)
        batch=next_steps.tensor.size(0)
        if next_steps.restored_step_index is not None:
            # index_start=next_steps.restored_step_index
            # index_end=(next_steps.restored_step_index+batch)%(self.num_old+self.num_new)
            index=next_steps.restored_step_index
            
        else:
            next_steps.indices
            index_start=self.tracking_index
            index_end=index_start+batch
            #print(index_end)
            if index_end>=self.num_old+self.num_new+self.num_fresh:
               
                index_start=self.num_old+self.num_new
                #print("tracking resetting to start")
                index_end=index_start+batch
            self.tracking_index=index_end
            index=torch.arange(index_start, index_end)
        #print("index",index.shape)
        n=index.size(0)
      
        sample_index=self.current_sample[index]
        # print(loss.shape,self.losses[index_start:index_end].shape)
        # print("shape",self.losses[index_start:index_end].shape,self.losses[index_start:index_end][sample_index].shape,self.losses[index_start:index_end,sample_index].shape)
        if loss.dim()==0:
            
            self.losses[index[:,None] ,sample_index]=loss
        else:
            self.losses[index[:,None],sample_index]=loss[:n]
        #print( self.softmax[index,sample_index].shape,next_steps.softmax[:n].shape)
        self.softmax[index]=next_steps.softmax[:n]
        self.indices[index]=next_steps.indices[:n]
        # self.losses[index_start:index_end]=loss[:n]
        self.occurrences[index]+=1
       
       
        self.current_sample[index]=(self.current_sample[index]+1)%self.num_samples
    @torch.no_grad()
    def reset_fresh(self):
        self.tracking_index=self.num_old+self.num_new
        self.occurrences[self.num_new+self.num_old:]=0
        self.current_sample[self.num_new+self.num_old:]=0
        self.losses[self.num_new+self.num_old:]=math.inf
    
       
    @torch.no_grad()
    def update(self):
        # indices=torch.argsort(self.get_efficiency(),descending=True)[self.num_old+self.num_new:]
        # indices=torch.argsort(self.occurrences[indices],descending=True)
        efficiency=self.get_efficiency()
        efficiency[:self.num_old]=math.inf
        sorted_indices_by_efficiency = torch.argsort(efficiency, descending=True)

        # Select the subset of items, skipping the top self.num_old+self.num_new items.
        subset_indices = sorted_indices_by_efficiency[self.num_old+self.num_new:]

        # Now, get the occurrences of these selected items.
        subset_occurrences = self.occurrences[subset_indices]

        # Finally, sort these selected items by their occurrences in descending order.
        # Note: We sort subset_occurrences, but we need to sort subset_indices based on these occurrences.
        sorted_indices_by_occurrences = subset_indices[torch.argsort(subset_occurrences, descending=True)]

        self.softmax[self.num_old+self.num_new:]=self.softmax[sorted_indices_by_occurrences]
        self.indices[self.num_old+self.num_new]=self.indices[sorted_indices_by_occurrences]
        self.losses[self.num_old+self.num_new]=self.losses[sorted_indices_by_occurrences]
        self.occurrences[self.num_old+self.num_new]=self.occurrences[sorted_indices_by_occurrences]
        self.current_sample[self.num_old+self.num_new]=self.current_sample[sorted_indices_by_occurrences]
        self.reset_fresh()
        
        
        
    def get_new(self,next_steps:NextSteps):
       # print("self.softmax.shape,next_steps.softmax.shape",self.softmax.shape,next_steps.softmax.shape)
       
        softmax=self.softmax[self.num_old:self.num_old+self.num_new].view(self.num_new,-1)
        next_steps_softmax=next_steps.softmax.view(next_steps.softmax.size(0),-1)
        with torch.no_grad():
            similarity=cosine_similarity_2d(softmax,next_steps_softmax).detach()
        
            index=torch.argmax(similarity,dim=0).detach()

       
        confidence=torch.gather(similarity,0,index.unsqueeze(0)).squeeze(0) *next_steps.confidence
        # print("softmax--",self.softmax[index].shape,next_steps.softmax.shape)
        # print("indices--",self.indices[index].shape,next_steps.indices.shape)
        # print("confidence",confidence.shape,next_steps.confidence.shape)
        
        return NextSteps(self.softmax[index],self.indices[index],self.softmax[index],None,confidence=confidence,restored_step_index=index)
    def get_old(self,next_steps:NextSteps):
        #print("self.softmax.shape,next_steps.softmax.shape",self.softmax.shape,next_steps.softmax.shape)
        softmax=self.softmax[:self.num_old].view(self.num_old,-1)
        next_steps_softmax=next_steps.softmax.view(next_steps.softmax.size(0),-1)
        with torch.no_grad():
            similarity=cosine_similarity_2d(softmax,next_steps_softmax).detach()
        
            index=torch.argmax(similarity,dim=0).detach()
        
        
        confidence=torch.gather(similarity ,0,index.unsqueeze(0)).squeeze(0)*next_steps.confidence
        # print("softmax--",self.softmax[index].shape,next_steps.softmax.shape)
        # print("indices--",self.indices[index].shape,next_steps.indices.shape)
        # print("confidence",confidence.shape,next_steps.confidence.shape)
        return NextSteps(self.softmax[index],self.indices[index],self.softmax[index],None,confidence=confidence,restored_step_index=index)
    def get_fresh(self,next_steps):
        return next_steps
    def get_random(self, next_steps):
        weight=list(self.old_new_fresh_distribution)
        #print(self.losses[0][0].item() ,self.losses[self.num_old][0].item() == math.inf)
        if self.losses[0][0].item() == math.inf:
            weight[0]=0
        if self.losses[self.num_old][0].item() == math.inf:
            weight[1]=0
        which=random.choices((self.get_old,self.get_new,self.get_fresh),weights=weight,k=1)[0]
        return which(next_steps)
        