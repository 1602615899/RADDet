import torch
from torch.utils.data import Sampler, Dataset
import random
from collections import defaultdict

class BucketBatchSampler(Sampler):
    """
    分桶批次采样器
    根据数据尺寸将样本分组到不同的桶中，每个批次只从同一个桶采样
    """
    def __init__(self, dataset, batch_size, bucket_boundaries, drop_last=False, shuffle=True):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            bucket_boundaries: 桶边界列表，例如 [50, 150, 300] 表示 3个桶
            drop_last: 是否丢弃不完整的批次
            shuffle: 是否打乱数据
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_boundaries = sorted(bucket_boundaries)
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 计算每个样本的尺寸（序列长度）
        self.sample_sizes = self._get_sample_sizes()
        
        # 将样本分配到桶中
        self.buckets = self._create_buckets()
        
        # 创建批次
        self.batches = self._create_batches()
        
    def _get_sample_sizes(self):
        """获取每个样本的尺寸"""
        sample_sizes = []
        for i in range(len(self.dataset)):
            # 通过dataset的__getitem__方法获取样本信息
            # 这里假设返回的是 (path_info, data_tensor)
            try:
                path_info, data_tensor = self.dataset[i]
                # 计算序列长度（总的元素数量）
                if hasattr(data_tensor, 'shape'):
                    size = data_tensor.numel()  # 总元素数
                else:
                    size = 1
                sample_sizes.append(size)
            except Exception as e:
                print(f"Warning: Could not get size for sample {i}: {e}")
                sample_sizes.append(1)
        return sample_sizes
    
    def _get_bucket_id(self, size):
        """根据尺寸确定桶ID"""
        for i, boundary in enumerate(self.bucket_boundaries):
            if size <= boundary:
                return i
        return len(self.bucket_boundaries)  # 最后一个桶
    
    def _create_buckets(self):
        """将样本分配到桶中"""
        buckets = defaultdict(list)
        for i, size in enumerate(self.sample_sizes):
            bucket_id = self._get_bucket_id(size)
            buckets[bucket_id].append(i)
        return buckets
    
    def _create_batches(self):
        """为每个桶创建批次"""
        batches = []
        for bucket_id, indices in self.buckets.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # 创建批次
            batch_indices = []
            for idx in indices:
                batch_indices.append(idx)
                if len(batch_indices) == self.batch_size:
                    batches.append(batch_indices)
                    batch_indices = []
            
            # 处理剩余的样本
            if batch_indices and not self.drop_last:
                batches.append(batch_indices)
        
        if self.shuffle:
            random.shuffle(batches)
        
        return batches
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

class SizeAwareBucketSampler(Sampler):
    """
    基于具体尺寸的分桶采样器
    直接根据数据的HxW或DxHxW尺寸分桶
    """
    def __init__(self, dataset, batch_size, size_buckets=None, drop_last=False, shuffle=True):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            size_buckets: 尺寸桶定义，例如 {0: (64,256,256), 1: (16,128,128), 2: (128,128,128)}
            drop_last: 是否丢弃不完整的批次
            shuffle: 是否打乱数据
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_buckets = size_buckets or self._default_buckets()
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 获取每个样本的尺寸
        self.sample_shapes = self._get_sample_shapes()
        
        # 将样本分配到桶中
        self.buckets = self._create_shape_buckets()
        
        # 创建批次
        self.batches = self._create_batches()
        
    def _default_buckets(self):
        """默认的尺寸桶"""
        return {
            0: (64, 256, 256),    # RADDet,CARRADA
            1: (16, 128, 128),    # CRUW
            2: (64, 128, 256),   # 
        }
    
    def _get_sample_shapes(self):
        """获取每个样本的形状"""
        sample_shapes = []
        for i in range(min(1000, len(self.dataset))):  # 只采样前1000个样本以提高效率
            try:
                path_info, data_tensor = self.dataset[i]
                if hasattr(data_tensor, 'shape'):
                    shape = tuple(data_tensor.shape)
                    # 如果是4D，取后3维；如果是3D，保持不变
                    if len(shape) == 4:
                        shape = shape[1:]  # 去掉batch维度
                    elif len(shape) == 5:
                        shape = shape[2:]  # 去掉batch和channel维度
                else:
                    shape = (1, 1, 1)
                sample_shapes.append(shape)
            except Exception as e:
                print(f"Warning: Could not get shape for sample {i}: {e}")
                sample_shapes.append((1, 1, 1))
        
        # 如果采样数量不足，补充剩余样本
        if len(sample_shapes) < len(self.dataset):
            remaining = len(self.dataset) - len(sample_shapes)
            sample_shapes.extend([sample_shapes[0]] * remaining)
            
        return sample_shapes
    
    def _find_closest_bucket(self, shape):
        """找到最接近的桶"""
        min_diff = float('inf')
        best_bucket_id = 0
        
        for bucket_id, bucket_shape in self.size_buckets.items():
            # 计算形状差异
            diff = sum(abs(s - b) for s, b in zip(shape[-3:], bucket_shape[-3:]))
            if diff < min_diff:
                min_diff = diff
                best_bucket_id = bucket_id
        
        return best_bucket_id
    
    def _create_shape_buckets(self):
        """根据形状将样本分配到桶中"""
        buckets = defaultdict(list)
        for i in range(len(self.sample_shapes)):
            shape = self.sample_shapes[i]
            bucket_id = self._find_closest_bucket(shape)
            buckets[bucket_id].append(i)
        return buckets
    
    def _create_batches(self):
        """为每个桶创建批次"""
        batches = []
        for bucket_id, indices in self.buckets.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # 创建批次
            batch_indices = []
            for idx in indices:
                batch_indices.append(idx)
                if len(batch_indices) == self.batch_size:
                    batches.append(batch_indices)
                    batch_indices = []
            
            # 处理剩余的样本
            if batch_indices and not self.drop_last:
                batches.append(batch_indices)
        
        if self.shuffle:
            random.shuffle(batches)
        
        return batches
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)