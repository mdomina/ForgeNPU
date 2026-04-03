module systolic_tile_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ACC_WIDTH = 32;
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int PE_COUNT = ROWS * COLS;

  logic signed [PE_COUNT*DATA_WIDTH-1:0] activations_i;
  logic signed [PE_COUNT*DATA_WIDTH-1:0] weights_i;
  logic signed [PE_COUNT*ACC_WIDTH-1:0] psums_i;
  logic compute_en;
  logic clear_acc;
  logic signed [PE_COUNT*ACC_WIDTH-1:0] psums_o;
  logic [PE_COUNT-1:0] valids_o;

  systolic_tile #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS)
  ) dut (
    .activations_i(activations_i),
    .weights_i(weights_i),
    .psums_i(psums_i),
    .compute_en(compute_en),
    .clear_acc(clear_acc),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  initial begin
    activations_i = {8'sd4, 8'sd3, 8'sd2, 8'sd1};
    weights_i = {8'sd8, 8'sd7, 8'sd6, 8'sd5};
    psums_i = {32'sd3, 32'sd2, 32'sd1, 32'sd0};
    compute_en = 1'b1;
    clear_acc = 1'b0;
    #1;
    if ($signed(psums_o[0 +: ACC_WIDTH]) !== 5 ||
        $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== 13 ||
        $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== 23 ||
        $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== 35 ||
        valids_o !== 4'b1111) begin
      $fatal(1, "Tile test 1 failed.");
    end

    activations_i = {8'sd4, 8'sd3, 8'sd2, 8'sd1};
    weights_i = {8'sd1, 8'sd1, 8'sd1, 8'sd1};
    psums_i = {32'sd40, 32'sd30, 32'sd20, 32'sd10};
    compute_en = 1'b0;
    clear_acc = 1'b0;
    #1;
    if ($signed(psums_o[0 +: ACC_WIDTH]) !== 10 ||
        $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== 20 ||
        $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== 30 ||
        $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== 40 ||
        valids_o !== 4'b0000) begin
      $fatal(1, "Tile test 2 failed.");
    end

    activations_i = {8'sd9, 8'sd9, 8'sd9, 8'sd9};
    weights_i = {8'sd2, 8'sd2, 8'sd2, 8'sd2};
    psums_i = {32'sd4, 32'sd3, 32'sd2, 32'sd1};
    compute_en = 1'b1;
    clear_acc = 1'b1;
    #1;
    if (psums_o !== '0 || valids_o !== 4'b0000) begin
      $fatal(1, "Tile test 3 failed.");
    end

    $display("systolic_tile_tb passed");
    $finish;
  end
endmodule
