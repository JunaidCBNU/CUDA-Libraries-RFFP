`timescale 1ns / 1ps
module Adder #(
    parameter RFFP_EXP_WIDTH = 8,
    parameter RFFP_MAN_WIDTH = 8,
    parameter IN_OUT_WIDTH = RFFP_EXP_WIDTH + RFFP_MAN_WIDTH
)(  
    
    input  logic signed [RFFP_EXP_WIDTH + RFFP_MAN_WIDTH:0] A,
    input  logic signed [RFFP_EXP_WIDTH + RFFP_MAN_WIDTH:0] B,
    output logic signed [IN_OUT_WIDTH:0] C

);

     logic sign_a;                
     logic [RFFP_EXP_WIDTH-1:0] exponent_a;
     logic [RFFP_MAN_WIDTH-1:0] mantissa_a;
     
     logic sign_b ;                          
     logic [RFFP_EXP_WIDTH-1:0] exponent_b;
     logic [RFFP_MAN_WIDTH-1:0] mantissa_b;
     logic result_sign;                        
     logic [RFFP_EXP_WIDTH-1:0] result_exponent;       
     logic [RFFP_MAN_WIDTH-1:0] result_mantissa;       

//    logic signed [RFFP_EXP_WIDTH-1:0] exponent_diff;
    logic [RFFP_EXP_WIDTH:0] abs_exponent_diff; 
    logic [RFFP_MAN_WIDTH-1:0] temp_mantissa_a,temp_mantissa_b ;
    logic [RFFP_EXP_WIDTH-1:0] adjusted_exponent;
    logic [RFFP_MAN_WIDTH:0] mantissa_add;
    

    assign sign_a          = A[IN_OUT_WIDTH];
    assign exponent_a      = A[IN_OUT_WIDTH-1: IN_OUT_WIDTH - RFFP_EXP_WIDTH];
    assign mantissa_a      = A[RFFP_MAN_WIDTH-1:0];
 
    assign sign_b          = B[IN_OUT_WIDTH];
    assign exponent_b      = B[IN_OUT_WIDTH-1: IN_OUT_WIDTH- RFFP_EXP_WIDTH];
    assign mantissa_b      = B[RFFP_MAN_WIDTH-1:0];
 

    // Adjust mantissas based on exponent difference
    always_comb begin
        if (exponent_a >= exponent_b) begin
            abs_exponent_diff = exponent_a - exponent_b;
            temp_mantissa_a = mantissa_a;
            temp_mantissa_b = mantissa_b >> abs_exponent_diff;
            adjusted_exponent = exponent_a;
        end else begin
            abs_exponent_diff = exponent_b - exponent_a;
            temp_mantissa_a = mantissa_a >> abs_exponent_diff;
            temp_mantissa_b = mantissa_b;
            adjusted_exponent = exponent_b;
        end
    end
    
    // Perform addition of mantissas based on sign
    always_comb begin
           
        if (sign_a == sign_b) begin
            result_sign = sign_a;
            mantissa_add = temp_mantissa_a + temp_mantissa_b;
        end
        else begin
            if (temp_mantissa_a >= temp_mantissa_b) begin
                result_sign = sign_a;
                mantissa_add = temp_mantissa_a - temp_mantissa_b;
            end
            else begin
                result_sign = sign_b;
                mantissa_add = temp_mantissa_b - temp_mantissa_a;     
            end

        end
    end
    
    // Normalize the mantissa
    always_comb begin
        if (exponent_a != exponent_b) begin
            result_mantissa = mantissa_add[RFFP_MAN_WIDTH-1 : 0] ;
            result_exponent = adjusted_exponent; 
        end
        else begin
            result_mantissa = mantissa_add[RFFP_MAN_WIDTH : 1];
            result_exponent = adjusted_exponent +1; 
            if (mantissa_add[0] ) begin
                result_mantissa = result_mantissa + 1;
            end
        end
    end

    always_comb begin  
        C[IN_OUT_WIDTH] = result_sign;
        C[IN_OUT_WIDTH-1-:RFFP_EXP_WIDTH] = result_exponent;
        C[RFFP_MAN_WIDTH-1:0] = result_mantissa;
    end
endmodule
