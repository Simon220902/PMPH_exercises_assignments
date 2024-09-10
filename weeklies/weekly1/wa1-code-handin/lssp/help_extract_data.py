# Assuming the data is stored in a list of strings
zeros_data_sequential = """test_data/lssp-zeros-data-small-interval-small.in:        279μs (95% CI: [     278.6,      278.9])
test_data/lssp-zeros-data-large-interval-small.in:        179μs (95% CI: [     178.7,      179.0])
test_data/lssp-zeros-data-small-interval-large.in:       2882μs (95% CI: [    2881.0,     2883.1])
test_data/lssp-zeros-data-large-interval-large.in:       1782μs (95% CI: [    1781.3,     1783.0])
test_data/lssp-zeros-data-small-interval-very-large.in:     165794μs (95% CI: [  165753.2,   165862.0])
test_data/lssp-zeros-data-large-interval-very-large.in:     109274μs (95% CI: [  103844.6,   119738.9])""".split("\n")

zeros_data_parallel = """test_data/lssp-zeros-data-small-interval-small.in:         36μs (95% CI: [      36.3,       36.5])
test_data/lssp-zeros-data-large-interval-small.in:         37μs (95% CI: [      36.4,       36.7])
test_data/lssp-zeros-data-small-interval-large.in:         60μs (95% CI: [      60.1,       60.2])
test_data/lssp-zeros-data-large-interval-large.in:         60μs (95% CI: [      60.2,       60.3])
test_data/lssp-zeros-data-small-interval-very-large.in:       1454μs (95% CI: [    1454.0,     1454.9])
test_data/lssp-zeros-data-large-interval-very-large.in:       1452μs (95% CI: [    1451.7,     1452.3])""".split("\n")


sorted_data_sequential = """test_data/lssp-sorted-data-small-interval-small.in:        459μs (95% CI: [     458.4,      459.0])
test_data/lssp-sorted-data-large-interval-small.in:        452μs (95% CI: [     451.9,      452.8])
test_data/lssp-sorted-data-small-interval-large.in:       4543μs (95% CI: [    4540.2,     4546.9])
test_data/lssp-sorted-data-large-interval-large.in:       4494μs (95% CI: [    4489.1,     4498.7])
test_data/lssp-sorted-data-small-interval-very-large.in:     258447μs (95% CI: [  258170.1,   258706.6])
test_data/lssp-sorted-data-large-interval-very-large.in:     255434μs (95% CI: [  255034.7,   256210.5])""".split("\n")
sorted_data_parallel = """test_data/lssp-sorted-data-small-interval-small.in:         36μs (95% CI: [      35.9,       36.1])
test_data/lssp-sorted-data-large-interval-small.in:         43μs (95% CI: [      42.5,       42.9])
test_data/lssp-sorted-data-small-interval-large.in:         62μs (95% CI: [      62.3,       62.5])
test_data/lssp-sorted-data-large-interval-large.in:         62μs (95% CI: [      61.9,       62.1])
test_data/lssp-sorted-data-small-interval-very-large.in:       1454μs (95% CI: [    1453.5,     1454.4])
test_data/lssp-sorted-data-large-interval-very-large.in:       1453μs (95% CI: [    1453.0,     1453.6])""".split("\n")


same_data_sequential = """test_data/lssp-sorted-data-small-interval-small.in:        212μs (95% CI: [     211.3,      212.2])
test_data/lssp-sorted-data-large-interval-small.in:        131μs (95% CI: [     130.4,      130.9])
test_data/lssp-sorted-data-small-interval-large.in:       2156μs (95% CI: [    2149.2,     2172.2])
test_data/lssp-sorted-data-large-interval-large.in:       1303μs (95% CI: [    1300.3,     1305.8])
test_data/lssp-sorted-data-small-interval-very-large.in:     162949μs (95% CI: [  142060.3,   186756.1])
test_data/lssp-sorted-data-large-interval-very-large.in:      75211μs (95% CI: [   75026.4,    75372.2])""".split("\n")
same_data_parallel = """test_data/lssp-sorted-data-small-interval-small.in:         36μs (95% CI: [      36.1,       36.3])
test_data/lssp-sorted-data-large-interval-small.in:         35μs (95% CI: [      34.9,       35.0])
test_data/lssp-sorted-data-small-interval-large.in:         59μs (95% CI: [      59.3,       59.4])
test_data/lssp-sorted-data-large-interval-large.in:         60μs (95% CI: [      59.8,       59.9])
test_data/lssp-sorted-data-small-interval-very-large.in:       1454μs (95% CI: [    1453.2,     1454.0])
test_data/lssp-sorted-data-large-interval-very-large.in:       1461μs (95% CI: [    1460.9,     1461.6])""".split("\n")

def make_dict(data):
    # Create a dictionary to store the results
    result_dict = {}

    for entry in data:
        # Split the entry to extract the file name and the time
        file_name, time_info = entry.split(':', 1)
        microseconds = int(time_info.split('μs')[0].strip())
        result_dict[file_name.strip()] = microseconds
    return result_dict

def make_speedup_dict(sequential, parallel):
    result_dict = {}
    for (filename, sequential_time) in sequential.items():
        parallel_time = parallel[filename]
        result_dict[filename] = sequential_time/parallel_time
    return result_dict

def print_speedup_dict(d, s):
    for (filename, speedup) in d.items():
        filename_trimmed = filename.replace(f"test_data/lssp-{s}-data", "")
        print(f"{filename_trimmed}: {round(speedup,2)}")

print("ZEROS:")
print_speedup_dict(make_speedup_dict(make_dict(zeros_data_sequential), make_dict(zeros_data_parallel)), "zeros")
print("SORTED:")
print_speedup_dict(make_speedup_dict(make_dict(sorted_data_sequential), make_dict(sorted_data_parallel)), "sorted")
print("SAME:")
print_speedup_dict(make_speedup_dict(make_dict(same_data_sequential), make_dict(same_data_parallel)), "sorted")