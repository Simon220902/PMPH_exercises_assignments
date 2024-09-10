-- Parallel Longest Satisfying Segment
--
-- ==
-- entry: main
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
-- compiled input {
--    [1i32, -2i32, -2i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    2i32
-- }
-- compiled input {
--    [1i32, 2i32, 3i32, 1i32, 2i32, 3i32]
-- }
-- output {
--    1i32
-- }
-- compiled input {
--    [1i32, -1i32, 1i32, 1i32, -1i32, -1i32, 1i32]
-- }
-- output {
--    2i32
-- }
-- compiled input @ test_data/lssp-sorted-data-small-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-small-interval-large.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-large.in
-- compiled input @ test_data/lssp-sorted-data-small-interval-very-large.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-very-large.in

import "lssp"
import "lssp-seq"

entry main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  -- in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

-- Sequential Longest Satisfying Segment
--
-- ==
-- entry: mainSeq
-- compiled input @ test_data/lssp-sorted-data-small-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-small-interval-large.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-large.in
-- compiled input @ test_data/lssp-sorted-data-small-interval-very-large.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-very-large.in
entry mainSeq (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  in  lssp_seq pred1 pred2 xs
  -- in  lssp pred1 pred2 xs