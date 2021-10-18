import os
import torch
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from torchvision import transforms, datasets
from PIL import Image

import argparse


# blur
def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)



# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        
    def evaluate(self, img_batch, exp_batch):
        r"""Efficiently evaluate big batch of images.Z
        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.
        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        preds = self.model(img_batch.cuda())
        preds = F.softmax(preds, dim=1).cpu().detach()
        predictions = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        substrate = self.substrate_fn(img_batch)

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in range(n_steps+1):
            # Compute new scores
            preds = self.model(start.cuda())
            preds = F.softmax(preds, dim=1).cpu().detach().numpy()
            scores[i] = preds[range(n_samples), top]
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().detach().numpy().reshape(n_samples, 3, HW)[r, :, coords] = finish.cpu().detach().numpy().reshape(n_samples, 3, HW)[r, :, coords]
#         print('AUC: {}'.format(auc(scores.mean(1))))
        return scores

class InterpretTransformer(object):
    def __init__(self, model, img_size=224):
        self.model = model
        self.model.eval()
        self.img_size=img_size
    
    def transition_attention_maps(self, input, index=None, start_layer=0, steps=20, with_integral=True, first_state=False):
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
    
        sal = F.interpolate(states[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()
    
    
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
            
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()
    
    
    def raw_attn(self, input, index=None):
        b = input.shape[0]
        output = self.model(input, register_hook=True)

        attrs = self.model.blocks[-1].attn.get_attention_map().mean(dim=1)
    
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()
    
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
        
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='insertion and deletion evaluation')
    parser.add_argument('--method', type=str,
            default='tam',
            choices=['tam',
                     'rollout',
                     'raw_attn',
                     'attribution'],
            help='')
    parser.add_argument('--batch_size', type=int,
                        default=16,
                        help='')
    parser.add_argument('--num_samples', type=int,
                        default=2000,
                        help='')
    parser.add_argument('--blur', action='store_true',
                        default=False,
                        help='')
    
    parser.add_argument('--arch', type=str,
            default='vit_base_patch16_224',
            choices=['vit_base_patch16_224',
                     'vit_base_patch16_384',
                     'vit_large_patch16_224',
                     'deit_base_patch16_224'],
            help='')
    
    args = parser.parse_args()
    
    if args.method in [
        'tam', 'raw_attn', 'rollout'
    ]:
        from baselines.ViT.ViT_new import vit_base_patch16_224, vit_large_patch16_224, deit_base_patch16_224, vit_base_patch16_384
        model = eval(args.arch)(pretrained=True).cuda()
    else:
        from baselines.ViT.ViT_LRP import vit_base_patch16_224, vit_large_patch16_224, deit_base_patch16_224, vit_base_patch16_384
        model = eval(args.arch)(pretrained=True).cuda()

    if args.arch == 'vit_base_patch16_384':
        img_size = 384
    else:
        img_size = 224
        
    HW = img_size * img_size 

    n_classes = 1000

    it = InterpretTransformer(model, img_size)
    
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    
    batch_size = args.batch_size
    num_samples = args.num_samples
    
    # blur
    if args.blur:
        print("use blur insertion")
        insertion = CausalMetric(model, 'ins', img_size * 8, substrate_fn=blur)
    else:
        print("use zero insertion")
        insertion = CausalMetric(model, 'ins', img_size * 8, substrate_fn=torch.zeros_like)
    
    deletion = CausalMetric(model, 'del', img_size * 8, substrate_fn=torch.zeros_like)

    scores = {'del': [], 'ins': []}

    dataset = datasets.ImageFolder('/root/datasets/ImageNet/val', preprocess)
    np.random.seed(0)
    max_index = np.random.randint(num_samples, len(dataset))
    print("subset indices: ", [max_index-num_samples, max_index])
    sub_dataset = torch.utils.data.Subset(dataset, indices=range(max_index-num_samples, max_index))
    
    # Load batch of images
    data_loader = torch.utils.data.DataLoader(
        sub_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    images = np.empty((len(data_loader), batch_size, 3, img_size, img_size))
    iterator = tqdm(data_loader, total=len(data_loader))
    for j, (img, _) in enumerate(iterator):
        if args.method == 'tam':
            exp = it.transition_attention_maps(img.cuda())
        elif args.method == 'raw_attn':
            exp = it.raw_attn(img.cuda()) 
        elif args.method == 'rollout':
            exp = it.rollout(img.cuda()) 
        elif args.method == 'attribution':
            exp = it.attribution(img.cuda())
        

        # Evaluate deletion
        h = deletion.evaluate(img, exp)
        scores['del'].append(auc(h.mean(1)))

        # Evaluate insertion
        h = insertion.evaluate(img, exp)
        scores['ins'].append(auc(h.mean(1)))
        iterator.set_description('del: %.4f, ins: %.4f' % (np.mean(scores['del']), np.mean(scores['ins'])))
        
    np.save(os.path.join('del_ins_results', args.method + '.npy'), scores)
    print('----------------------------------------------------------------')
    print('Final:\nDeletion - {:.5f}\nInsertion - {:.5f}'.format(np.mean(scores['del']), np.mean(scores['ins'])))