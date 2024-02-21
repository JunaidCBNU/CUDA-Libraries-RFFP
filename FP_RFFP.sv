`timescale 1ns / 1ps
module FP_RFFP #(
    parameter FP_WIDTH = 16,
    parameter EXP_WIDTH = 8,
    parameter MANTISSA_WIDTH = 7,

    parameter RFFP_EXP_WIDTH = 8,
    parameter RFFP_MAN_WIDTH = 7,
    parameter EXPONENT_BIAS = 75
)(
    input logic signed [FP_WIDTH - 1:0] number,
    output logic signed [RFFP_EXP_WIDTH + RFFP_MAN_WIDTH:0] result_binary
);
    
    logic signed  signs;
    logic signed [RFFP_EXP_WIDTH - 1:0] exponents, exponent_shifted;
    logic signed [MANTISSA_WIDTH - 1:0] mantissas;
    logic signed [RFFP_MAN_WIDTH - 1:0] mantissa_rounded, mantissa_shifted;
    
    logic round_bits;
    logic round_conditions;
    
    always_comb begin
        // Extract components from the input number
        signs = number[FP_WIDTH - 1];
        exponents = number[FP_WIDTH -2: MANTISSA_WIDTH];
        mantissas = number[MANTISSA_WIDTH - 1:0];

        // Calculate shifted exponents
        if (RFFP_EXP_WIDTH == 6) begin 
            exponent_shifted = (exponents != {RFFP_EXP_WIDTH{1'b0}}) ? (exponents - EXPONENT_BIAS + (7-RFFP_MAN_WIDTH)) : {RFFP_EXP_WIDTH{1'b0}};
        end
        else begin
            exponent_shifted = (exponents !={RFFP_EXP_WIDTH{1'b0}}) ? (exponents - (128 - (2**(RFFP_EXP_WIDTH-1))) + (7-RFFP_MAN_WIDTH)) : {RFFP_EXP_WIDTH{1'b0}};
        end

        // Calculate shifted mantissas
       // mantissa_explicit_1 = (exponents != {EXP_WIDTH{1'b0}}) ? {1'b1,mantissas} : {1'd0,mantissas};
        mantissa_shifted = mantissas >> ((7 - RFFP_MAN_WIDTH));
        
        // Calculate round bits and conditions
//        round_bits = mantissas[7 - RFFP_MAN_WIDTH -1];
        round_bits =0;
        round_conditions = (round_bits == 1) && (mantissa_shifted != {RFFP_MAN_WIDTH{1'b1}});

        // Update shifted mantissas based on round conditions
        mantissa_rounded = (round_conditions) ? (mantissa_shifted + 1) : mantissa_shifted;

        // Concatenate components to form the result_binary
        result_binary = {signs, exponent_shifted, mantissa_rounded};
    end

endmodule
