# Copyright 2026 Individual Contributor: donghyq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def classify_native_history(rows, prompt_length, output_length, overlap_enabled, finished):
    """Separate output-producing forwards from one discarded overlap tail."""
    productive, discarded = [], []
    for row in rows:
        start, end = row["output_interval"]
        assert end >= start, "Native output interval regressed"
        (productive if end > start else discarded).append(row)
    assert len(productive) == output_length, "Missing or extra output-producing forward"
    assert [row["output_interval"] for row in productive] == [[i, i + 1] for i in range(output_length)], (
        "Output intervals repeated, dropped, or reordered"
    )
    expected = prompt_length + output_length - 1
    assert [p for row in productive for p in row["positions"]] == list(range(expected)), (
        "History repeated, dropped, or unexpected prefix reuse"
    )
    assert len(discarded) <= 1, "Multiple non-output forwards are unexplained"
    if discarded:
        tail = discarded[0]
        assert overlap_enabled and finished, "Non-output forward outside finished overlap"
        assert rows[-1] is tail, "Non-output forward is not the terminal tail"
        assert tail["output_interval"] == [output_length, output_length]
        assert tail["positions"] == [expected], "Tail repeated history or skipped an input"
        assert tail["forward_id"] > productive[-1]["forward_id"], "Tail forward order invalid"
    return productive, discarded
