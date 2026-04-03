module mac_unit #(
  parameter int DATA_WIDTH = 8,
  parameter int ACC_WIDTH = 32
) (
  input  logic signed [DATA_WIDTH-1:0] a,
  input  logic signed [DATA_WIDTH-1:0] b,
  input  logic signed [ACC_WIDTH-1:0] acc_in,
  output logic signed [ACC_WIDTH-1:0] acc_out
);
  logic signed [2*DATA_WIDTH-1:0] product;

  always_comb begin
    product = a * b;
    acc_out = acc_in + product;
  end
endmodule
