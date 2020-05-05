import random
import time
from collections import namedtuple
from typing import List

import pandas as pd

SOS_CHAR = "$"
EOS_CHAR = "^"

Data = namedtuple(
    "Data", ["encoder_input", "decoder_input", "decoder_target", "sequence_lengths"]
)


class StrToIntConverter:
    def __init__(self, *lists_of_strs: List[str]):
        vocab = sorted(
            set(
                char
                for list_of_strs in lists_of_strs
                for string in list_of_strs
                for char in string
            )
        )
        # use 0 as a padding index
        shifted_index_lookup = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.index_lookup = shifted_index_lookup

    def convert_all(self, rows: List[str]):
        return [[self.index_lookup[char] for char in string] for string in rows]


class DataGenerator:
    def __init__(self):
        self._converter = None

    @property
    def converter(self):
        if not self._converter:
            raise AttributeError("Must call generate() in order to set the converter")
        return self._converter

    @converter.setter
    def converter(self, converter):
        self._converter = converter

    @property
    def vocab_size(self):
        return len(self.converter.index_lookup)

    def generate(self, m: int) -> Data:
        """
        :param m: number of rows to generate
        :return:
        """
        # TODO: does the type matter?
        curr_time_in_seconds = int(time.time())
        random_times_since_epoch = [
            time.localtime(random.randrange(0, curr_time_in_seconds)) for _ in range(m)
        ]
        x_eng_raw = [
            time.strftime("%B %d, %Y", random_time)
            for random_time in random_times_since_epoch
        ]
        x_eng_seq_lengths = pd.Series([len(e) for e in x_eng_raw])

        x_robo_raw = [
            time.strftime("%Y-%m-%d", random_time)
            for random_time in random_times_since_epoch
        ]

        self.converter = StrToIntConverter(x_eng_raw, x_robo_raw)

        encoder_input = (
            pd.DataFrame(self.converter.convert_all(x_eng_raw))
            .fillna(value=0)
            .astype("int32")
        )
        decoder_input = pd.DataFrame(
            self.converter.convert_all([SOS_CHAR + row for row in x_robo_raw])
        ).astype("int32")
        # TODO: don't concatenate strings directly! could be slow
        decoder_target = pd.DataFrame(
            self.converter.convert_all([row + EOS_CHAR for row in x_robo_raw])
        ).astype("int32")

        return Data(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            decoder_target=decoder_target,
            sequence_lengths=x_eng_seq_lengths,  # TODO: is this what it should be??
        )


if __name__ == "__main__":
    generator = DataGenerator()
    data = generator.generate(200)
    print(generator.vocab_size)
    print(data.encoder_input.head())
