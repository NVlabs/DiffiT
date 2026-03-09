import pickle

import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import numpy as np


FORCE_DEPLOY = False
global bias_indx
bias_indx = -1

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda
        self.grid_exists = False

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape

            if not(self.grid_exists)\
                    or input_tensor.shape[0] != self.col_idx_v.shape[0] \
                    or input_tensor.shape[2] != self.col_idx_v.shape[2] \
                    or input_tensor.shape[3] != self.col_idx_v.shape[3] \
                    or input_tensor.device != self.col_idx_v.device:
                # print(f"here {input_tensor.shape}")
                n_col, n_row = dim_y, dim_x
                col_vals = torch.arange(0, n_col)
                col_repeat = col_vals.repeat(n_row, 1)
                col_idx = col_repeat.view(1, 1, n_row, n_col)

                row_vals = torch.arange(0, n_row).view(n_row, -1)
                row_repeat = row_vals.repeat(1, n_col)
                row_idx = row_repeat.view(1, 1, n_row, n_col)

                self.col_idx_v = col_idx.float().to(input_tensor.device)/float(n_col-1)*2 - 1
                self.row_idx_v = row_idx.float().to(input_tensor.device)/float(n_row-1)*2 - 1

                self.col_idx_v = torch.repeat_interleave(self.col_idx_v, batch_size_shape, 0)
                self.row_idx_v = torch.repeat_interleave(self.row_idx_v, batch_size_shape, 0)

                self.col_idx_v.detach_()
                self.row_idx_v.detach_()

                self.grid_exists = True

            out = torch.cat([input_tensor, self.col_idx_v, self.row_idx_v], dim=1)
        elif self.rank == 1:
            batch_size_shape, dim_x, channel_in_shape = input_tensor.shape

            if not(self.grid_exists) \
                    or input_tensor.shape[0] != self.row_idx_v.shape[0] \
                    or input_tensor.shape[1] != self.row_idx_v.shape[1] \
                    or input_tensor.shape[2] != self.row_idx_v.shape[2] \
                    or input_tensor.device != self.row_idx_v.device:

                # print(f"here {input_tensor.shape}")
                n_row =  dim_x

                row_vals = torch.arange(0, n_row).view(1,  n_row, 1)
                self.row_idx_v = row_vals.float().to(input_tensor.device)/float(n_row-1)*2 - 1
                # self.row_idx_v = self.row_idx_v.repeat(batch_size_shape, 1, 1)

                self.row_idx_v.detach_()

                self.grid_exists = True

            # out = torch.cat([input_tensor, self.row_idx_v], dim=2)
            out = self.row_idx_v

        else:
            raise NotImplementedError

        return out


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, with_r=False, use_cuda=True):

        super().__init__()

        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

class CoordConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, with_r=False, use_cuda=True):

        super().__init__()

        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        # self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
        #                       kernel_size, stride, padding, dilation, groups, bias)
        # self.conv = nn.Linear(in_channels + self.rank + int(with_r), out_channels,
        #                       bias=bias)
        self.conv = nn.Linear(self.rank + int(with_r), out_channels,
                              bias=bias)

        self.grid_exists = False
        self.pos_emb = None


    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """

        if self.training:
            self.grid_exists = False

        if self.grid_exists:
            if self.pos_emb.shape[1] != input_tensor.shape[1] or input_tensor.device != self.pos_emb.device:
                self.grid_exists = False

        if not self.grid_exists:
            out = self.addcoords(input_tensor)
            pos_emb = self.conv(out)
            self.pos_emb = pos_emb
            self.grid_exists = True
        else:
            pos_emb = self.pos_emb

        out = input_tensor + pos_emb

        return out



class PosEmbMLPSwinv2D(nn.Module):
    def __init__(self, window_size, pretrained_window_size, num_heads, seq_length, ct_correct=False, no_log=False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        # carrier_token
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.grid_exists = False
        self.pos_emb = None

        self.deploy = False

        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)

        self.ct_correct=ct_correct

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor, local_window_size):


        if self.deploy or FORCE_DEPLOY:
            input_tensor += self.relative_bias
            return input_tensor
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            # print(input_tensor.shape, local_window_size, self.relative_coords_table.shape)
            if 0:
                global bias_indx
                bias_indx += 1
                kernels = self.cpb_mlp(self.relative_coords_table)
                print(kernels.shape, input_tensor.shape, local_window_size)
                with open(f"temp_kernels/kernel_{bias_indx:04d}.pkl", "wb") as f:
                    pickle.dump(kernels[0].type(torch.float32).data.cpu().numpy(), f)
                        # matplotlib.pyplot.imshow(X
                # exit()
                if bias_indx>35:
                    exit()

            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            n_global_feature = input_tensor.shape[2] - local_window_size
            if n_global_feature>0 and self.ct_correct:

                step_for_ct=self.window_size[0]/(n_global_feature**0.5+1)
                seq_length = int(n_global_feature ** 0.5)
                indices = []
                for i in range(seq_length):
                    for j in range(seq_length):
                        ind = (i+1)*step_for_ct*self.window_size[0] + (j+1)*step_for_ct
                        indices.append(int(ind))

                top_part = relative_position_bias[:, indices, :]
                lefttop_part = relative_position_bias[:, indices, :][:,:,indices]
                left_part = relative_position_bias[:, :, indices]

            # print(relative_position_bias.shape)
            relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature,
                                                                                      0,
                                                                                      n_global_feature,
                                                                                      0)).contiguous()
            if n_global_feature>0 and self.ct_correct:
                # add positional bias between local and CT
                relative_position_bias = relative_position_bias*0.0
                relative_position_bias[:,:n_global_feature,:n_global_feature] = lefttop_part
                relative_position_bias[:,:n_global_feature,n_global_feature:] = top_part
                relative_position_bias[:,n_global_feature:,:n_global_feature] = left_part

            self.pos_emb = relative_position_bias.unsqueeze(0)

            self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor



class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,  dim, rank=2, seq_length=4, conv=False):
        super().__init__()
        self.rank = rank
        # mlp to generate continuous relative position bias
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1,bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1,bias=False))
        self.grid_exists = False
        self.pos_emb = None

        self.deploy = False

        relative_bias = torch.zeros(1,seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        # print(relative_bias.shape)
        self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]

        if self.deploy or FORCE_DEPLOY:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False

        # if self.grid_exists and input_tensor.device != self.pos_emb.device:
        #     self.grid_exists = False
        # need to handle shape mismatch to set

        if not self.grid_exists:
            self.grid_exists = True
            if self.rank==1:
                # 1d sequence
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
                relative_coords_h -= seq_length//2
                relative_coords_h /= (seq_length//2)
                relative_coords_table = relative_coords_h

                self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
                self.relative_bias = self.pos_emb
            else:
                # input is a vector, but it is a patch if unrolled, therefore 2D encoding
                seq_length = int(seq_length**0.5)
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
                relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
                relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)

                relative_coords_table -= seq_length // 2
                relative_coords_table /= (seq_length // 2)
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1,2))
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                # if (self.pos_emb.shape != self.relative_bias.shape):
                #     print(self.pos_emb.shape != self.relative_bias.shape)
                self.relative_bias = self.pos_emb

        # print(self.relative_bias.shape)
        input_tensor = input_tensor + self.pos_emb

        return input_tensor
