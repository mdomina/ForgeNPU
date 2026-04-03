module tile_compute_unit #(
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
  input  logic dma_valid_i,
  input  logic dma_write_weights_i,
  input  logic [ADDR_WIDTH-1:0] dma_addr_i,
  input  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_i,
  input  logic load_vector_en_i,
  input  logic [ADDR_WIDTH-1:0] activation_read_addr_i,
  input  logic [ADDR_WIDTH-1:0] weight_read_addr_i,
  input  logic compute_en_i,
  input  logic clear_acc_i,
  output logic scratchpad_vector_valid_o,
  output logic dma_done_o,
  output logic dma_busy_o,
  output logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_o,
  output logic [ROWS*COLS-1:0] valids_o
);
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north;
  logic write_activations_en;
  logic [ADDR_WIDTH-1:0] activation_write_addr;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data;
  logic write_weights_en;
  logic [ADDR_WIDTH-1:0] weight_write_addr;
  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data;

  dma_engine #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dma_inst (
    .clk(clk),
    .rst_n(rst_n),
    .dma_valid_i(dma_valid_i),
    .dma_write_weights_i(dma_write_weights_i),
    .dma_addr_i(dma_addr_i),
    .dma_payload_i(dma_payload_i),
    .write_activations_en_o(write_activations_en),
    .activation_write_addr_o(activation_write_addr),
    .activations_write_data_o(activations_write_data),
    .write_weights_en_o(write_weights_en),
    .weight_write_addr_o(weight_write_addr),
    .weights_write_data_o(weights_write_data),
    .dma_done_o(dma_done_o),
    .dma_busy_o(dma_busy_o)
  );

  scratchpad_controller #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) scratchpad_inst (
    .clk(clk),
    .rst_n(rst_n),
    .write_activations_en_i(write_activations_en),
    .activation_write_addr_i(activation_write_addr),
    .activations_write_data_i(activations_write_data),
    .write_weights_en_i(write_weights_en),
    .weight_write_addr_i(weight_write_addr),
    .weights_write_data_i(weights_write_data),
    .load_vector_en_i(load_vector_en_i),
    .activation_read_addr_i(activation_read_addr_i),
    .weight_read_addr_i(weight_read_addr_i),
    .activations_west_o(activations_west),
    .weights_north_o(weights_north),
    .vector_valid_o(scratchpad_vector_valid_o)
  );

  systolic_tile #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS)
  ) tile_inst (
    .clk(clk),
    .rst_n(rst_n),
    .activations_west_i(activations_west),
    .weights_north_i(weights_north),
    .load_inputs_en(scratchpad_vector_valid_o),
    .compute_en(compute_en_i),
    .clear_acc(clear_acc_i),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );
endmodule
