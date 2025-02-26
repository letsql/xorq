import itertools
import traceback

import pyarrow as pa
import pyarrow.compute as pc
import toolz

import xorq as xo


@toolz.curry
def excepts_print_exc(func, exc=Exception, handler=toolz.functoolz.return_none):
    _handler = toolz.compose(handler, toolz.curried.do(traceback.print_exception))
    return toolz.excepts(exc, func, _handler)


def copy_rbr_batches(reader):
    manager = pa.default_cpu_memory_manager()
    gen = (batch.copy_to(manager) for batch in reader)
    return pa.RecordBatchReader.from_batches(reader.schema, gen)


def make_filtered_reader(reader):
    gen = (chunk.data for chunk in reader if chunk.data)
    return pa.RecordBatchReader.from_batches(reader.schema, gen)


def instrument_reader(reader, prefix=""):
    from xorq.common.utils.logging_utils import get_print_logger

    logger = get_print_logger()

    def gen(reader):
        logger.info(f"{prefix:10s}first batch yielded")
        yield next(reader)
        yield from reader
        logger.info(f"{prefix:10s}last batch yielded")

    return pa.RecordBatchReader.from_batches(reader.schema, gen(reader))


@excepts_print_exc
def streaming_split_exchange(
    split_key, f, context, reader, writer, options=None, **kwargs
):
    started = False
    g = excepts_print_exc(f)
    for split_reader in ReaderSplitter(
        make_filtered_reader(reader),
        split_key,
    ):
        batch = g(split_reader)
        if not started:
            writer.begin(batch.schema, options=options)
            started = True
        writer.write_batch(batch)


class ReaderSplitter:
    """Yield multiple record batch readers from a single record batch reader by `split_key`"""

    def __init__(self, rbr, split_key):
        self.rbr = rbr
        self.split_key = split_key
        self._curr_batch = toolz.excepts(StopIteration, next)(self.rbr)

    @property
    def curr_batch(self):
        if not self._curr_batch or not len(self._curr_batch):
            return
        return self._curr_batch

    @property
    def curr_key(self):
        if not self.curr_batch:
            raise ValueError
        (curr_key,) = self.curr_batch[self.split_key].slice(length=1).tolist()
        return curr_key

    def split_batch(self, batch, curr_key):
        condition = pc.field(self.split_key) == curr_key
        try:
            good_batch = batch.filter(condition)
        except IndexError:
            return (None, batch)
        if len(good_batch) == len(batch):
            return (batch, None)
        else:
            bad_batch = batch.filter(~condition)
            return (good_batch, bad_batch)

    @property
    def next_split(self):
        batches = ()
        curr_key = self.curr_key
        (curr_batch, self._curr_batch) = (self.curr_batch, None)
        for batch in itertools.chain([curr_batch], self.rbr):
            good_batch, bad_batch = self.split_batch(batch, curr_key)
            if good_batch:
                batches += (good_batch,)
            if bad_batch:
                self._curr_batch = bad_batch
                return pa.RecordBatchReader.from_batches(
                    curr_batch.schema,
                    batches,
                )
        # fall through condition
        return pa.RecordBatchReader.from_batches(
            curr_batch.schema,
            batches,
        )

    def __next__(self):
        return self.next_split

    def __iter__(self):
        while self.curr_batch:
            yield self.next_split

    @classmethod
    def gen_splits(cls, expr, split_key):
        rbr = xo.to_pyarrow_batches(expr.order_by(split_key))
        yield from cls(rbr, split_key)
