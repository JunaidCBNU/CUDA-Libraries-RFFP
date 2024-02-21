`timescale 1ns / 1ps
module RFFP_FP_MUL #(
    parameter EXP_WIDTH = 8,
    parameter MAN_WIDTH = 7,
    parameter RFFP_EXP  = 6,
    parameter RFFP_MAN_WIDTH = 8,
    parameter IN_OUT_WIDTH = RFFP_EXP + RFFP_MAN_WIDTH
)(

    input logic [IN_OUT_WIDTH:0] input_RFFP,
    output logic [EXP_WIDTH + MAN_WIDTH:0] output_FP
);

    logic sign_RFFP;
    logic [RFFP_EXP-1:0] exponent_RFFP;
    logic [RFFP_MAN_WIDTH-1:0] mantissa_RFFP;

    logic [3:0] shift_mul;
    logic [RFFP_MAN_WIDTH-1:0] mantissa_FP;
    logic [EXP_WIDTH-1:0] exponent_FP;
    int i;

    assign sign_RFFP          = input_RFFP[IN_OUT_WIDTH];
    assign exponent_RFFP      = input_RFFP[IN_OUT_WIDTH-1: IN_OUT_WIDTH - RFFP_EXP];
    assign mantissa_RFFP      = input_RFFP[RFFP_MAN_WIDTH-1:0];
 

    // Calculate shift_mul using Verilog operations
    always_comb begin
        shift_mul = 0;
        for (i = RFFP_MAN_WIDTH-1; i >= 0; i = i - 1) begin
            if (mantissa_RFFP[i] == 1'b1) begin
                break;
            end
            shift_mul = shift_mul + 1;
        end

        if (shift_mul >= RFFP_MAN_WIDTH) begin
            shift_mul = 0;
        end
    end

    // Left shift the mantissa and handle overflow
    always_comb begin
        mantissa_FP = mantissa_RFFP << 1;
    end

    // Calculate exponent_FP using Verilog operations
    always_comb begin
        if (RFFP_EXP == 6) begin
            if ((exponent_RFFP != 0)&& (mantissa_RFFP!=0)) begin
                exponent_FP = exponent_RFFP - shift_mul + 76;
            end 
            else begin
                exponent_FP = {EXP_WIDTH{1'b0}};
            end
        end else begin
            if ((exponent_RFFP != 0) && (mantissa_RFFP!=0)) begin
                exponent_FP = exponent_RFFP - shift_mul + 1 +  128 - (2**(EXP_WIDTH-1)) ;
//                 exponent_FP = exponent_RFFP + 1  ;
            end 
            else begin
                exponent_FP = {EXP_WIDTH{1'b0}};
            end
        end
    end

    // assign mantissa_mul_shift = {mantissa_FP, 3'b0};
    // Create the EXP_WIDTH + RFFP_MAN_WIDTH - 1 bit binary representation
    always_comb begin
        output_FP = {sign_RFFP, exponent_FP, mantissa_FP[RFFP_MAN_WIDTH-1:1]};
    end

endmodule
