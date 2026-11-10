import re


def parse_register_file(filename):
    reg_map = {}
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r'#define\s+(\S+_Address)\s+0x([0-9A-Fa-f]{1,8})', line)
            if match:
                reg_name = match.group(1)
                reg_address = int(match.group(2), 16)
                reg_map[reg_address] = reg_name

    # 调试用：查看某个地址是否被正确解析
    # for address,name in reg_map.items():
    #     if address == 0x00018280:
    #         print("============")
    #         print("address:0x{:08X},name: {}".format(address,name))
    #         #print("address:0x{:08X},name: {}".format(address,name))
    return reg_map


def parse_mutil_reg_file(filenames):
    complete_reg_map = {}
    for filename in filenames:
        reg_map = parse_register_file(filename)
        complete_reg_map.update(reg_map)
    return complete_reg_map


def process_capture_file(reg_map, capture_filename, output_filename):
    with open(capture_filename, 'r') as capture_file, open(output_filename, 'w') as output_file:
        for line in capture_file:
            match = re.match(r"(CFG_APB_(RD|WR))\s+'h([0-9A-Fa-f]+)\s+'h([0-9A-Fa-f]{1,8})", line)
            if match:
                operation = match.group(1)
                address = int(match.group(3), 16)
                data = match.group(4)

                reg_name = reg_map.get(address)
                if reg_name:
                    output_line = "{} 'h{:08X} {:2} #{} \n".format(operation, address, data, reg_name)
                else:
                    output_line = "{} 'h{:08X} 'h{:2}\n".format(operation, address, data)
                output_file.write(output_line)
            else:
                output_file.write(line)


def main():
    print("######## search reg ########")

    ### Need to replace ###
    fe0_reg_file = "/workspace/FPGA_DPU/home/wanghualei/test_drm/driver/verisilicon/9x00/vs_dc_reg_fe0.h"
    fe1_reg_file = "/workspace/FPGA_DPU/home/wanghualei/test_drm/driver/verisilicon/9x00/vs_dc_reg_fe1.h"
    be_reg_file = "/workspace/FPGA_DPU/home/wanghualei/test_drm/driver/verisilicon/9x00/vs_dc_reg_be.h"
    capture_file = "/workspace/FPGA_DPU/home/wanghualei/test_drm/0723_pass.txt"
    output_file = "/workspace/FPGA_DPU/home/wanghualei/test_drm/0723_pass_reg.txt"
    ########################

    reg_file = [fe0_reg_file, fe1_reg_file, be_reg_file]
    reg_map = parse_mutil_reg_file(reg_file)

    process_capture_file(reg_map, capture_file, output_file)

    print("#### reg match finish... ####")


if __name__ == "__main__":
    main()
