module top_npu #(
  parameter int DATA_WIDTH = 8,
  parameter int ACC_WIDTH = 32,
  parameter int ROWS = 2,
  parameter int COLS = 2,
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int MAX_DIM = (ROWS > COLS) ? ROWS : COLS
) (
  input  logic clk,
  input  logic rst_n,
  input  logic start_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i,
  output logic busy_o,
  output logic done_o,
  output logic [3:0] scheduler_state_o,
  output logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_o,
  output logic [ROWS*COLS-1:0] valids_o
);
  logic dma_valid;
  logic dma_write_weights;
  logic [ADDR_WIDTH-1:0] dma_addr;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload;
  logic load_vector_en;
  logic [ADDR_WIDTH-1:0] activation_read_addr;
  logic [ADDR_WIDTH-1:0] weight_read_addr;
  logic compute_en;
  logic clear_acc;
  logic scratchpad_vector_valid_unused;
  logic dma_done_unused;
  logic dma_busy_unused;

  scheduler #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) scheduler_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start_i(start_i),
    .activation_slot0_i(activation_slot0_i),
    .activation_slot1_i(activation_slot1_i),
    .weight_slot0_i(weight_slot0_i),
    .weight_slot1_i(weight_slot1_i),
    .dma_valid_o(dma_valid),
    .dma_write_weights_o(dma_write_weights),
    .dma_addr_o(dma_addr),
    .dma_payload_o(dma_payload),
    .load_vector_en_o(load_vector_en),
    .activation_read_addr_o(activation_read_addr),
    .weight_read_addr_o(weight_read_addr),
    .compute_en_o(compute_en),
    .clear_acc_o(clear_acc),
    .busy_o(busy_o),
    .done_o(done_o),
    .state_o(scheduler_state_o)
  );

  tile_compute_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) tile_compute_inst (
    .clk(clk),
    .rst_n(rst_n),
    .dma_valid_i(dma_valid),
    .dma_write_weights_i(dma_write_weights),
    .dma_addr_i(dma_addr),
    .dma_payload_i(dma_payload),
    .load_vector_en_i(load_vector_en),
    .activation_read_addr_i(activation_read_addr),
    .weight_read_addr_i(weight_read_addr),
    .compute_en_i(compute_en),
    .clear_acc_i(clear_acc),
    .scratchpad_vector_valid_o(scratchpad_vector_valid_unused),
    .dma_done_o(dma_done_unused),
    .dma_busy_o(dma_busy_unused),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );
endmodule
