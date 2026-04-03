module scheduler_tb;
  localparam int DATA_WIDTH = 8;
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int MAX_DIM = (ROWS > COLS) ? ROWS : COLS;
  localparam logic [3:0] S_IDLE = 4'd0;
  localparam logic [3:0] S_DMA_ACT = 4'd1;
  localparam logic [3:0] S_DMA_WGT = 4'd2;
  localparam logic [3:0] S_LOAD = 4'd3;
  localparam logic [3:0] S_COMPUTE = 4'd4;
  localparam logic [3:0] S_CLEAR = 4'd5;
  localparam logic [3:0] S_DONE = 4'd6;

  logic clk;
  logic rst_n;
  logic start_i;
  logic [1:0] slot_count_i;
  logic [1:0] load_iterations_i;
  logic [3:0] compute_iterations_i;
  logic clear_on_done_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i;
  logic dma_valid_o;
  logic dma_write_weights_o;
  logic [ADDR_WIDTH-1:0] dma_addr_o;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_o;
  logic load_vector_en_o;
  logic [ADDR_WIDTH-1:0] activation_read_addr_o;
  logic [ADDR_WIDTH-1:0] weight_read_addr_o;
  logic compute_en_o;
  logic clear_acc_o;
  logic busy_o;
  logic done_o;
  logic [3:0] state_o;

  scheduler #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start_i(start_i),
    .slot_count_i(slot_count_i),
    .load_iterations_i(load_iterations_i),
    .compute_iterations_i(compute_iterations_i),
    .clear_on_done_i(clear_on_done_i),
    .activation_slot0_i(activation_slot0_i),
    .activation_slot1_i(activation_slot1_i),
    .weight_slot0_i(weight_slot0_i),
    .weight_slot1_i(weight_slot1_i),
    .dma_valid_o(dma_valid_o),
    .dma_write_weights_o(dma_write_weights_o),
    .dma_addr_o(dma_addr_o),
    .dma_payload_o(dma_payload_o),
    .load_vector_en_o(load_vector_en_o),
    .activation_read_addr_o(activation_read_addr_o),
    .weight_read_addr_o(weight_read_addr_o),
    .compute_en_o(compute_en_o),
    .clear_acc_o(clear_acc_o),
    .busy_o(busy_o),
    .done_o(done_o),
    .state_o(state_o)
  );

  always #5 clk = ~clk;

  task automatic step_and_expect(
    input logic start_value,
    input logic [3:0] expected_state,
    input logic expected_busy,
    input logic expected_done,
    input logic expected_dma_valid,
    input logic expected_dma_write_weights,
    input integer expected_dma_addr,
    input integer expected_payload0,
    input integer expected_payload1,
    input logic expected_load_vector,
    input integer expected_activation_read_addr,
    input integer expected_weight_read_addr,
    input logic expected_compute,
    input logic expected_clear
  );
    begin
      start_i = start_value;
      @(posedge clk);
      #1;
      if (state_o !== expected_state ||
          busy_o !== expected_busy ||
          done_o !== expected_done ||
          dma_valid_o !== expected_dma_valid ||
          dma_write_weights_o !== expected_dma_write_weights ||
          dma_addr_o !== expected_dma_addr[ADDR_WIDTH-1:0] ||
          $signed(dma_payload_o[0 +: DATA_WIDTH]) !== expected_payload0 ||
          $signed(dma_payload_o[DATA_WIDTH +: DATA_WIDTH]) !== expected_payload1 ||
          load_vector_en_o !== expected_load_vector ||
          activation_read_addr_o !== expected_activation_read_addr[ADDR_WIDTH-1:0] ||
          weight_read_addr_o !== expected_weight_read_addr[ADDR_WIDTH-1:0] ||
          compute_en_o !== expected_compute ||
          clear_acc_o !== expected_clear) begin
        $fatal(1, "scheduler_tb failed");
      end
      start_i = 1'b0;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start_i = 1'b0;
    slot_count_i = 2'd2;
    load_iterations_i = 2'd2;
    compute_iterations_i = 4'd2;
    clear_on_done_i = 1'b1;
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

    step_and_expect(1'b1, S_DMA_ACT, 1'b1, 1'b0, 1'b1, 1'b0, 0, 1, 2, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_DMA_WGT, 1'b1, 1'b0, 1'b1, 1'b1, 0, 5, 6, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_DMA_ACT, 1'b1, 1'b0, 1'b1, 1'b0, 1, 3, 4, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_DMA_WGT, 1'b1, 1'b0, 1'b1, 1'b1, 1, 7, 8, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_LOAD, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b1, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_LOAD, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b1, 1, 1, 1'b0, 1'b0);
    step_and_expect(1'b0, S_COMPUTE, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b1, 1'b0);
    step_and_expect(1'b0, S_COMPUTE, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b1, 1'b0);
    step_and_expect(1'b0, S_CLEAR, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b0, 1'b1);
    step_and_expect(1'b0, S_DONE, 1'b0, 1'b1, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_IDLE, 1'b0, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b0, 1'b0);

    slot_count_i = 2'd1;
    load_iterations_i = 2'd2;
    compute_iterations_i = 4'd1;
    clear_on_done_i = 1'b0;

    step_and_expect(1'b1, S_DMA_ACT, 1'b1, 1'b0, 1'b1, 1'b0, 0, 1, 2, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_DMA_WGT, 1'b1, 1'b0, 1'b1, 1'b1, 0, 5, 6, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_LOAD, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b1, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_LOAD, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b1, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_COMPUTE, 1'b1, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b1, 1'b0);
    step_and_expect(1'b0, S_DONE, 1'b0, 1'b1, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b0, 1'b0);
    step_and_expect(1'b0, S_IDLE, 1'b0, 1'b0, 1'b0, 1'b0, 0, 0, 0, 1'b0, 0, 0, 1'b0, 1'b0);

    $display("scheduler_tb passed");
    $finish;
  end
endmodule
