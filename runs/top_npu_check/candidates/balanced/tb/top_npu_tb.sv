module top_npu_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ACC_WIDTH = 32;
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int PE_COUNT = ROWS * COLS;
  localparam logic [3:0] S_IDLE = 4'd0;
  localparam logic [3:0] S_DMA_ACT0 = 4'd1;
  localparam logic [3:0] S_DMA_WGT0 = 4'd2;
  localparam logic [3:0] S_DMA_ACT1 = 4'd3;
  localparam logic [3:0] S_DMA_WGT1 = 4'd4;
  localparam logic [3:0] S_LOAD0 = 4'd5;
  localparam logic [3:0] S_LOAD1 = 4'd6;
  localparam logic [3:0] S_COMPUTE0 = 4'd7;
  localparam logic [3:0] S_COMPUTE1 = 4'd8;
  localparam logic [3:0] S_CLEAR = 4'd9;
  localparam logic [3:0] S_DONE = 4'd10;

  logic clk;
  logic rst_n;
  logic start_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i;
  logic busy_o;
  logic done_o;
  logic [3:0] scheduler_state_o;
  logic signed [PE_COUNT*ACC_WIDTH-1:0] psums_o;
  logic [PE_COUNT-1:0] valids_o;

  top_npu #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start_i(start_i),
    .activation_slot0_i(activation_slot0_i),
    .activation_slot1_i(activation_slot1_i),
    .weight_slot0_i(weight_slot0_i),
    .weight_slot1_i(weight_slot1_i),
    .busy_o(busy_o),
    .done_o(done_o),
    .scheduler_state_o(scheduler_state_o),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  always #5 clk = ~clk;

  task automatic step_and_expect(
    input logic start_value,
    input logic [3:0] expected_state,
    input logic expected_busy,
    input logic expected_done,
    input integer p0,
    input integer p1,
    input integer p2,
    input integer p3,
    input logic [PE_COUNT-1:0] expected_valids
  );
    begin
      start_i = start_value;
      @(posedge clk);
      #1;
      if (scheduler_state_o !== expected_state ||
          busy_o !== expected_busy ||
          done_o !== expected_done ||
          $signed(psums_o[0 +: ACC_WIDTH]) !== p0 ||
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== p1 ||
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== p2 ||
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== p3 ||
          valids_o !== expected_valids) begin
        $fatal(1, "top_npu_tb failed");
      end
      start_i = 1'b0;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start_i = 1'b0;
    activation_slot0_i = '0;
    activation_slot1_i = '0;
    weight_slot0_i = '0;
    weight_slot1_i = '0;
    activation_slot0_i[0 +: DATA_WIDTH] = 1;
    activation_slot0_i[DATA_WIDTH +: DATA_WIDTH] = 2;
    activation_slot1_i[0 +: DATA_WIDTH] = 3;
    activation_slot1_i[DATA_WIDTH +: DATA_WIDTH] = 4;
    weight_slot0_i[0 +: DATA_WIDTH] = 5;
    weight_slot0_i[DATA_WIDTH +: DATA_WIDTH] = 6;
    weight_slot1_i[0 +: DATA_WIDTH] = 7;
    weight_slot1_i[DATA_WIDTH +: DATA_WIDTH] = 8;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    step_and_expect(1'b1, S_DMA_ACT0, 1'b1, 1'b0, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_DMA_WGT0, 1'b1, 1'b0, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_DMA_ACT1, 1'b1, 1'b0, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_DMA_WGT1, 1'b1, 1'b0, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_LOAD0, 1'b1, 1'b0, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_LOAD1, 1'b1, 1'b0, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_COMPUTE0, 1'b1, 1'b0, 21, 8, 20, 12, 4'b1111);
    step_and_expect(1'b0, S_COMPUTE1, 1'b1, 1'b0, 42, 16, 40, 24, 4'b1111);
    step_and_expect(1'b0, S_CLEAR, 1'b1, 1'b0, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_DONE, 1'b0, 1'b1, 0, 0, 0, 0, 4'b0000);
    step_and_expect(1'b0, S_IDLE, 1'b0, 1'b0, 0, 0, 0, 0, 4'b0000);

    $display("top_npu_tb passed");
    $finish;
  end
endmodule
