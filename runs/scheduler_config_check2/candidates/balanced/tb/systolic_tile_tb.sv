module systolic_tile_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ACC_WIDTH = 32;
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int PE_COUNT = ROWS * COLS;

  logic clk;
  logic rst_n;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west_i;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north_i;
  logic load_inputs_en;
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
    .clk(clk),
    .rst_n(rst_n),
    .activations_west_i(activations_west_i),
    .weights_north_i(weights_north_i),
    .load_inputs_en(load_inputs_en),
    .compute_en(compute_en),
    .clear_acc(clear_acc),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  always #5 clk = ~clk;

  task automatic set_edge_inputs(
    input integer row0_act,
    input integer row1_act,
    input integer col0_weight,
    input integer col1_weight
  );
    begin
      activations_west_i = '0;
      weights_north_i = '0;
      activations_west_i[0 +: DATA_WIDTH] = row0_act;
      activations_west_i[DATA_WIDTH +: DATA_WIDTH] = row1_act;
      weights_north_i[0 +: DATA_WIDTH] = col0_weight;
      weights_north_i[DATA_WIDTH +: DATA_WIDTH] = col1_weight;
    end
  endtask

  task automatic expect_outputs(
    input integer p0,
    input integer p1,
    input integer p2,
    input integer p3,
    input logic [PE_COUNT-1:0] expected_valids
  );
    begin
      if ($signed(psums_o[0 +: ACC_WIDTH]) !== p0 ||
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== p1 ||
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== p2 ||
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== p3 ||
          valids_o !== expected_valids) begin
        $fatal(
          1,
          "Tile expectation failed: expected (%0d %0d %0d %0d %0b), got (%0d %0d %0d %0d %0b)",
          p0, p1, p2, p3, expected_valids,
          $signed(psums_o[0 +: ACC_WIDTH]),
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]),
          valids_o
        );
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    activations_west_i = '0;
    weights_north_i = '0;
    load_inputs_en = 1'b0;
    compute_en = 1'b0;
    clear_acc = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    set_edge_inputs(1, 2, 5, 6);
    load_inputs_en = 1'b1;
    compute_en = 1'b0;
    clear_acc = 1'b0;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    set_edge_inputs(3, 4, 7, 8);
    load_inputs_en = 1'b1;
    compute_en = 1'b0;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    set_edge_inputs(0, 0, 0, 0);
    load_inputs_en = 1'b0;
    compute_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(21, 8, 20, 12, 4'b1111);

    load_inputs_en = 1'b0;
    compute_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(42, 16, 40, 24, 4'b1111);

    compute_en = 1'b0;
    clear_acc = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    clear_acc = 1'b0;
    set_edge_inputs(9, 10, 1, 2);
    load_inputs_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    load_inputs_en = 1'b0;
    compute_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(9, 0, 0, 0, 4'b1111);

    $display("systolic_tile_tb passed");
    $finish;
  end
endmodule
