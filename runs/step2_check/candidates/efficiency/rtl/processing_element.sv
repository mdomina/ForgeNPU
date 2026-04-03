module processing_element #(
  parameter int DATA_WIDTH = 8,
  parameter int ACC_WIDTH = 32
) (
  input  logic signed [DATA_WIDTH-1:0] activation_i,
  input  logic signed [DATA_WIDTH-1:0] weight_i,
  input  logic signed [ACC_WIDTH-1:0] psum_i,
  input  logic compute_en,
  input  logic clear_acc,
  output logic signed [ACC_WIDTH-1:0] psum_o,
  output logic valid_o
);
  logic signed [ACC_WIDTH-1:0] mac_result;

  mac_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
  ) mac_inst (
    .a(activation_i),
    .b(weight_i),
    .acc_in(psum_i),
    .acc_out(mac_result)
  );

  always_comb begin
    if (clear_acc) begin
      psum_o = '0;
      valid_o = 1'b0;
    end else if (compute_en) begin
      psum_o = mac_result;
      valid_o = 1'b1;
    end else begin
      psum_o = psum_i;
      valid_o = 1'b0;
    end
  end
endmodule
