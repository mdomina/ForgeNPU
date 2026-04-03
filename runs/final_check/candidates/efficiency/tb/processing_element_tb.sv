module processing_element_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ACC_WIDTH = 32;

  logic signed [DATA_WIDTH-1:0] activation_i;
  logic signed [DATA_WIDTH-1:0] weight_i;
  logic signed [ACC_WIDTH-1:0] psum_i;
  logic compute_en;
  logic clear_acc;
  logic signed [ACC_WIDTH-1:0] psum_o;
  logic valid_o;

  processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
  ) dut (
    .activation_i(activation_i),
    .weight_i(weight_i),
    .psum_i(psum_i),
    .compute_en(compute_en),
    .clear_acc(clear_acc),
    .psum_o(psum_o),
    .valid_o(valid_o)
  );

  initial begin
    activation_i = 3;
    weight_i = 4;
    psum_i = 5;
    compute_en = 1'b1;
    clear_acc = 1'b0;
    #1;
    if (psum_o !== 17 || valid_o !== 1'b1) begin
      $fatal(1, "PE test 1 failed: expected (17, 1), got (%0d, %0d)", psum_o, valid_o);
    end

    activation_i = -2;
    weight_i = 5;
    psum_i = 11;
    compute_en = 1'b0;
    clear_acc = 1'b0;
    #1;
    if (psum_o !== 11 || valid_o !== 1'b0) begin
      $fatal(1, "PE test 2 failed: expected (11, 0), got (%0d, %0d)", psum_o, valid_o);
    end

    activation_i = 7;
    weight_i = 8;
    psum_i = 99;
    compute_en = 1'b1;
    clear_acc = 1'b1;
    #1;
    if (psum_o !== 0 || valid_o !== 1'b0) begin
      $fatal(1, "PE test 3 failed: expected (0, 0), got (%0d, %0d)", psum_o, valid_o);
    end

    $display("processing_element_tb passed");
    $finish;
  end
endmodule
