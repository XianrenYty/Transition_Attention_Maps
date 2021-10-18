import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class InterpretTransformer(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def transition_attention_maps(self, input, index=None, start_layer=4, steps=20, with_integral=True, first_state=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        for i in range(start_layer, num_blocks-1)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            states = states.bmm(attn)
            states += states_

        total_gradients = torch.zeros(b, h, s, s).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients

        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        states = states * W_state
        
        return states[:, 0, 1:]
    
    
    def attribution(self, input, index=None, start_layer=0):
        b = input.shape[0]
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).cuda(), **kwargs)

        b, h, s, _ = self.model.blocks[-1].attn.get_attn_gradients().shape

        num_blocks = len(self.model.blocks)
        # first_block
        attn = self.model.blocks[start_layer].attn.get_attn_cam()
        grad = self.model.blocks[start_layer].attn.get_attn_gradients()
        attr = (grad * attn).clamp(min=0).mean(1)
        # add residual
        eye = torch.eye(s).expand(b, s, s).cuda()
        attr = attr + eye
        attr = attr / attr.sum(dim=-1, keepdim=True)
        
        attrs = attr
        for i in range(start_layer+1, num_blocks):
            attn = self.model.blocks[i].attn.get_attn_cam()
            grad = self.model.blocks[i].attn.get_attn_gradients()
            attr = (grad * attn).clamp(min=0).mean(1)
            # add residual
            eye = torch.eye(s).expand(b, s, s).cuda()
            attr = attr + eye
            attr = attr / attr.sum(dim=-1, keepdim=True)
            
            attrs = attr.bmm(attrs)
        
        return attrs[:, 0, 1:]
    
    def raw_attn(self, input, index=None):
        b = input.shape[0]
        output = self.model(input, register_hook=True)

        attr = self.model.blocks[-1].attn.get_attention_map().mean(dim=1)
    
        return attr[:, 0, 1:]
    
    def rollout(self, input, index=None, start_layer=0):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attn_gradients().shape

        num_blocks = len(self.model.blocks)
        attrs = torch.eye(s).expand(b, h, s, s).cuda()
        for i in range(start_layer, num_blocks):
            attr = self.model.blocks[i].attn.get_attention_map()

            eye = torch.eye(s).expand(b, h, s, s).cuda()
            attr = attr + eye
            attr = attr / attr.sum(dim=-1, keepdim=True)

            attrs = (attr @ attrs)

        attrs = attrs.mean(1)
        
        return attrs[:, 0, 1:]