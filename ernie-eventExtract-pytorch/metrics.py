# import six
# import abc
import numpy as np

import torch


# @six.add_metaclass(abc.ABCMeta)
class Metric(object):
    r"""
    Base class for metric, encapsulates metric logic and APIs
    Usage:

        .. code-block:: text

            m = SomeMetric()
            for prediction, label in ...:
                m.update(prediction, label)
            m.accumulate()
        
    Advanced usage for :code:`compute`:

    Metric calculation can be accelerated by calculating metric states
    from model outputs and labels by build-in operators not by Python/NumPy
    in :code:`compute`, metric states will be fetched as NumPy array and
    call :code:`update` with states in NumPy format.
    Metric calculated as follows (operations in Model and Metric are
    indicated with curly brackets, while data nodes not):

        .. code-block:: text

                 inputs & labels              || ------------------
                       |                      ||
                    {model}                   ||
                       |                      ||
                outputs & labels              ||
                       |                      ||    tensor data
                {Metric.compute}              ||
                       |                      ||
              metric states(tensor)           ||
                       |                      ||
                {fetch as numpy}              || ------------------
                       |                      ||
              metric states(numpy)            ||    numpy data
                       |                      ||
                {Metric.update}               \/ ------------------

    Examples:

        For :code:`Accuracy` metric, which takes :code:`pred` and :code:`label`
        as inputs, we can calculate the correct prediction matrix between
        :code:`pred` and :code:`label` in :code:`compute`.
        For examples, prediction results contains 10 classes, while :code:`pred`
        shape is [N, 10], :code:`label` shape is [N, 1], N is mini-batch size,
        and we only need to calculate accurary of top-1 and top-5, we could
        calculate the correct prediction matrix of the top-5 scores of the
        prediction of each sample like follows, while the correct prediction
        matrix shape is [N, 5].

          .. code-block:: text

              def compute(pred, label):
                  # sort prediction and slice the top-5 scores
                  pred = paddle.argsort(pred, descending=True)[:, :5]
                  # calculate whether the predictions are correct
                  correct = pred == label
                  return paddle.cast(correct, dtype='float32')

        With the :code:`compute`, we split some calculations to OPs (which
        may run on GPU devices, will be faster), and only fetch 1 tensor with
        shape as [N, 5] instead of 2 tensors with shapes as [N, 10] and [N, 1].
        :code:`update` can be define as follows:

          .. code-block:: text

              def update(self, correct):
                  accs = []
                  for i, k in enumerate(self.topk):
                      num_corrects = correct[:, :k].sum()
                      num_samples = len(correct)
                      accs.append(float(num_corrects) / num_samples)
                      self.total[i] += num_corrects
                      self.count[i] += num_samples
                  return accs
    """

    def __init__(self):
        pass

    # @abc.abstractmethod
    def reset(self):
        """
        Reset states and result
        """
        raise NotImplementedError("function 'reset' not implemented in {}.".
                                  format(self.__class__.__name__))

    # @abc.abstractmethod
    def update(self, *args):
        """
        Update states for metric

        Inputs of :code:`update` is the outputs of :code:`Metric.compute`,
        if :code:`compute` is not defined, the inputs of :code:`update`
        will be flatten arguments of **output** of mode and **label** from data:
        :code:`update(output1, output2, ..., label1, label2,...)`

        see :code:`Metric.compute`
        """
        raise NotImplementedError("function 'update' not implemented in {}.".
                                  format(self.__class__.__name__))

    # @abc.abstractmethod
    def accumulate(self):
        """
        Accumulates statistics, computes and returns the metric value
        """
        raise NotImplementedError(
            "function 'accumulate' not implemented in {}.".format(
                self.__class__.__name__))

    # @abc.abstractmethod
    def name(self):
        """
        Returns metric name
        """
        raise NotImplementedError("function 'name' not implemented in {}.".
                                  format(self.__class__.__name__))

    def compute(self, *args):
        """
        This API is advanced usage to accelerate metric calculating, calulations
        from outputs of model to the states which should be updated by Metric can
        be defined here, where Paddle OPs is also supported. Outputs of this API
        will be the inputs of "Metric.update".

        If :code:`compute` is defined, it will be called with **outputs**
        of model and **labels** from data as arguments, all outputs and labels
        will be concatenated and flatten and each filed as a separate argument
        as follows:
        :code:`compute(output1, output2, ..., label1, label2,...)`

        If :code:`compute` is not defined, default behaviour is to pass
        input to output, so output format will be:
        :code:`return output1, output2, ..., label1, label2,...`

        see :code:`Metric.update`
        """
        return args


class Accuracy(Metric):
    """
    Encapsulates accuracy metric logic.

    Args:
        topk (int|tuple(int)): Number of top elements to look at
            for computing accuracy. Default is (1,).
        name (str, optional): String name of the metric instance. Default
            is `acc`.

    Example by standalone:
        
        .. code-block:: python

          import numpy as np
          import paddle

          x = paddle.to_tensor(np.array([
              [0.1, 0.2, 0.3, 0.4],
              [0.1, 0.4, 0.3, 0.2],
              [0.1, 0.2, 0.4, 0.3],
              [0.1, 0.2, 0.3, 0.4]]))
          y = paddle.to_tensor(np.array([[0], [1], [2], [3]]))

          m = paddle.metric.Accuracy()
          correct = m.compute(x, y)
          m.update(correct)
          res = m.accumulate()
          print(res) # 0.75


    Example with Model API:
        
        .. code-block:: python

          import paddle
          from paddle.static import InputSpec
          import paddle.vision.transforms as T
          from paddle.vision.datasets import MNIST
             
          input = InputSpec([None, 1, 28, 28], 'float32', 'image')
          label = InputSpec([None, 1], 'int64', 'label')
          transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
          train_dataset = MNIST(mode='train', transform=transform)

          model = paddle.Model(paddle.vision.LeNet(), input, label)
          optim = paddle.optimizer.Adam(
              learning_rate=0.001, parameters=model.parameters())
          model.prepare(
              optim,
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

          model.fit(train_dataset, batch_size=64)

    """

    def __init__(self, topk=(1, ), name=None, *args, **kwargs):
        super(Accuracy, self).__init__(*args, **kwargs)
        self.topk = topk
        self.maxk = max(topk)
        self._init_name(name)
        self.reset()

    def compute(self, pred, label, *args):
        """
        Compute the top-k (maxinum value in `topk`) indices.

        Args:
            pred (Tensor): The predicted value is a Tensor with dtype
                float32 or float64. Shape is [batch_size, d0, ..., dN].
            label (Tensor): The ground truth value is Tensor with dtype
                int64. Shape is [batch_size, d0, ..., 1], or
                [batch_size, d0, ..., num_classes] in one hot representation.
                
        Return:
            Tensor: Correct mask, a tensor with shape [batch_size, topk].
        """
        pred = torch.argsort(pred, descending=True)
        if len(pred.shape) == 2:
            pred = pred[:, 0:self.maxk]
        elif len(pred.shape) == 3:
            pred = pred[:, :, 0:self.maxk]
        elif len(pred.shape) == 1:
            pred = pred[0:self.maxk]

        if (len(label.shape) == 1) or \
           (len(label.shape) == 2 and label.shape[-1] == 1):
            # In static mode, the real label data shape may be different
            # from shape defined by paddle.static.InputSpec in model
            # building, reshape to the right shape.
            label = torch.reshape(label, (-1, 1))
        elif label.shape[-1] != 1:
            # one-hot label
            label = torch.argmax(label, dim=-1, keepdim=True)
        correct = pred == label
        return correct.float()

    def update(self, correct, *args):
        """
        Update the metrics states (correct count and total count), in order to
        calculate cumulative accuracy of all instances. This function also
        returns the accuracy of current step.
        
        Args:
            correct: Correct mask, a tensor with shape [batch_size, topk].

        Return:
            Tensor: the accuracy of current step.
        """
        if isinstance(correct, torch.Tensor):
            correct = correct.numpy()
        num_samples = np.prod(np.array(correct.shape[:-1]))
        accs = []
        for i, k in enumerate(self.topk):
            num_corrects = correct[..., :k].sum()
            accs.append(float(num_corrects) / num_samples)
            self.total[i] += num_corrects
            self.count[i] += num_samples
        accs = accs[0] if len(self.topk) == 1 else accs
        return accs

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.total = [0.] * len(self.topk)
        self.count = [0] * len(self.topk)

    def accumulate(self):
        """
        Computes and returns the accumulated metric.
        """
        res = []
        for t, c in zip(self.total, self.count):
            r = float(t) / c if c > 0 else 0.
            res.append(r)
        res = res[0] if len(self.topk) == 1 else res
        return res

    def _init_name(self, name):
        name = name or 'acc'
        if self.maxk != 1:
            self._name = ['{}_top{}'.format(name, k) for k in self.topk]
        else:
            self._name = [name]

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name


if __name__ == "__main__":
    pred = torch.Tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.4, 0.3, 0.2],
        [0.1, 0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3, 0.4]
    ])
    label = torch.Tensor([[0], [1], [2], [3]])
    print("pred: ", pred)
    metric = Accuracy()
    res = metric.compute(pred, label)
    acc = metric.update(res)
    accu = metric.accumulate()
    print("res: ", res)
    print("acc: ", acc)
    print("accu: ", accu)