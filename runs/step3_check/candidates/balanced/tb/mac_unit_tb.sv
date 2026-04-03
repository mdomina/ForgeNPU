module mac_unit_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ACC_WIDTH = 32;

  logic signed [DATA_WIDTH-1:0] a;
  logic signed [DATA_WIDTH-1:0] b;
  logic signed [ACC_WIDTH-1:0] acc_in;
  logic signed [ACC_WIDTH-1:0] acc_out;

  mac_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
  ) dut (
    .a(a),
    .b(b),
    .acc_in(acc_in),
    .acc_out(acc_out)
  );

  initial begin
    a = 4;
    b = 3;
    acc_in = 2;
    #1;
    if (acc_out !== 14) begin
      $fatal(1, "Test 1 failed: expected 14, got %0d", acc_out);
    end

    a = -5;
    b = 6;
    acc_in = 10;
    #1;
    if (acc_out !== -20) begin
      $fatal(1, "Test 2 failed: expected -20, got %0d", acc_out);
    end

    $display("mac_unit_tb passed");
    $finish;
  end
endmodule
