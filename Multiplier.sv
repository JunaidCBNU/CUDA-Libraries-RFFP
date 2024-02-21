`timescale 1ns / 1ps
module Multiplier #(
    parameter RFFP_EXP_WIDTH = 8,
    parameter RFFP_MAN_WIDTH = 7,
    parameter IN_OUT_WIDTH = RFFP_EXP_WIDTH + RFFP_MAN_WIDTH + 1
)(

    input wire [RFFP_EXP_WIDTH + RFFP_MAN_WIDTH:0] A,
    input wire [RFFP_EXP_WIDTH + RFFP_MAN_WIDTH:0] B,
    output logic signed [IN_OUT_WIDTH:0] C

);
    logic number_sign  ;                    
    logic [RFFP_EXP_WIDTH-1:0] number_exp;
    logic [RFFP_MAN_WIDTH-1:0] number_mantissa ;

    logic number_b_sign;
    logic [RFFP_EXP_WIDTH-1:0] number_b_exp;
    logic [RFFP_MAN_WIDTH-1:0] number_b_mantissa;

    logic result_sign;
    logic [RFFP_EXP_WIDTH-1:0] result_exp;
    logic [RFFP_MAN_WIDTH:0] result_mantissa;


    logic [(RFFP_MAN_WIDTH+1)*2-1:0] mantissa_mul, result_man_normalized;
    logic signed [RFFP_EXP_WIDTH:0] exp_mul;
    logic [RFFP_EXP_WIDTH-1:0] bias = (RFFP_EXP_WIDTH == 6 ? 52 :(1 << (RFFP_EXP_WIDTH - 1)) - 1);
    logic [3:0] shift_amount;
    logic [RFFP_MAN_WIDTH:0] mantissa_explicit_a, mantissa_explicit_b;
    logic [RFFP_EXP_WIDTH-1:0] max_exp = (1 << (RFFP_EXP_WIDTH) - 1);
    
    always_comb begin
        number_sign = A[RFFP_EXP_WIDTH + RFFP_MAN_WIDTH];
        number_exp = A[RFFP_EXP_WIDTH + RFFP_MAN_WIDTH-1 -: RFFP_EXP_WIDTH];
        number_mantissa = A[RFFP_MAN_WIDTH-1:0];

        number_b_sign = B[RFFP_EXP_WIDTH + RFFP_MAN_WIDTH];
        number_b_exp = B[RFFP_EXP_WIDTH + RFFP_MAN_WIDTH-1 -: RFFP_EXP_WIDTH];
        number_b_mantissa = B[RFFP_MAN_WIDTH-1:0];
    end

    assign mantissa_explicit_a = {1'b1, number_mantissa}; 
    assign mantissa_explicit_b = {1'b1, number_b_mantissa};

    // XOR operation for sign
    assign result_sign = number_sign ^ number_b_sign;

    // Element-wise multiplication for mantissa
    assign mantissa_mul = mantissa_explicit_a * mantissa_explicit_b;

    // Determine how much to shift to normalize the mantissa
    int i;
    always_comb begin
        shift_amount = 0;
        for (i = (RFFP_MAN_WIDTH+1)*2-1; i >= 0; i = i - 1) begin
            if (mantissa_mul[i] == 1'b1) begin
                break;
            end
            else begin
                shift_amount = shift_amount + 1;
            end
        end
        result_man_normalized = mantissa_mul << shift_amount;
        result_mantissa = result_man_normalized[(RFFP_MAN_WIDTH+1)*2-1 : RFFP_MAN_WIDTH +1];
        if (result_man_normalized[(RFFP_MAN_WIDTH+1)*2 -2 - RFFP_MAN_WIDTH] ) begin
            result_mantissa = result_mantissa + 1;   
        end
        if (number_exp == 0 || number_b_exp == 0 || exp_mul < 0 || exp_mul > max_exp) 
            result_mantissa = 0; // Handle special case for zero exponent
    end

    // Calculate the exponent
  always_comb begin
        if (number_exp == 0 || number_b_exp == 0) 
            exp_mul = 0; // Handle special case for zero exponent
        else
           exp_mul = number_exp + number_b_exp - bias - shift_amount ;
        // Handle exponent overflow or underflow
        if (exp_mul > max_exp) 
            result_exp = max_exp; // Cap at max exponent value
        else if (exp_mul < 0) 
            result_exp = 0; // Underflow to zero
        else 
            result_exp = exp_mul[RFFP_EXP_WIDTH-1:0];
    end

    always_comb begin  
        C[IN_OUT_WIDTH] = result_sign;
        C[IN_OUT_WIDTH-1-:RFFP_EXP_WIDTH] = result_exp;
        C[RFFP_MAN_WIDTH:0] = result_mantissa;
    end
endmodule
