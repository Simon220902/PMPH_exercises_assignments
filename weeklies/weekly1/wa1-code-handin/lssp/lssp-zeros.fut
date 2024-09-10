-- Parallel Longest Satisfying Segment
--
-- ==
-- entry: main
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
-- compiled input {
--    [0i32, 0, -2, 1, 0, 1, 0, 0, -3, 0, 0, 0, 1]
-- }
-- output {
--    3
-- }
-- compiled input {
--    [-3i32, -2, -1, 0, 1, 2, 3]
-- }
-- output {
--    1
-- }
-- compiled input {
--    [-3i32, -2, -1, 1, 2, 3]
-- }
-- output {
--    0
-- }
-- compiled input @ test_data/lssp-zeros-data-small-interval-small.in
-- compiled input @ test_data/lssp-zeros-data-large-interval-small.in
-- compiled input @ test_data/lssp-zeros-data-small-interval-large.in
-- compiled input @ test_data/lssp-zeros-data-large-interval-large.in

-- INPUTS:
-- futhark dataset --i32-bounds=-1:1 -b -g '[64000]i32' > test_data/lssp-zeros-data-small-interval-small.in
-- futhark dataset --i32-bounds=-2040:2040 -b -g '[64000]i32' > test_data/lssp-zeros-data-large-interval-small.in
-- futhark dataset --i32-bounds=-1:1 -b -g '[64000000]i32' > test_data/lssp-zeros-data-small-interval-large.in
-- futhark dataset --i32-bounds=-2040:2040 -b -g '[64000000]i32' > test_data/lssp-zeros-data-large-interval-large.in

import "lssp-seq"
import "lssp"

type int = i32

entry main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0) in
  -- lssp_seq pred1 pred2 xs
  lssp pred1 pred2 xs

-- Sequential Longest Satisfying Segment
--
-- ==
-- entry: mainSeq
-- compiled input @ test_data/lssp-zeros-data-small-interval-small.in
-- compiled input @ test_data/lssp-zeros-data-large-interval-small.in
-- compiled input @ test_data/lssp-zeros-data-small-interval-large.in
-- compiled input @ test_data/lssp-zeros-data-large-interval-large.in
entry mainSeq (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0) in
  lssp_seq pred1 pred2 xs
  -- lssp pred1 pred2 xs